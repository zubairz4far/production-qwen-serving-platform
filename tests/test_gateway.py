from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from gateway.config import Settings
from gateway.main import create_app


def settings(**overrides) -> Settings:
    values = {
        "upstream_base_url": "http://upstream",
        "upstream_api_key": "private-upstream",
        "public_api_keys": "public-a,public-b",
        "allowed_models": "tool-calling,base",
        "readiness_model": "tool-calling",
        "max_concurrency": 2,
        "queue_timeout_seconds": 0.05,
    }
    values.update(overrides)
    return Settings(**values)


def test_liveness_does_not_require_auth() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    with TestClient(create_app(settings(), transport=transport)) as client:
        response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_missing_public_api_key() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    with TestClient(create_app(settings(), transport=transport)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "tool-calling", "messages": []},
        )
    assert response.status_code == 401


def test_gateway_never_forwards_callers_bearer_token() -> None:
    seen_authorization: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_authorization.append(request.headers["authorization"])
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    with TestClient(create_app(settings(), transport=transport)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer public-a"},
            json={"model": "tool-calling", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert seen_authorization == ["Bearer private-upstream"]


def test_model_allowlist_blocks_unapproved_model() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    with TestClient(create_app(settings(), transport=transport)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer public-a"},
            json={"model": "other-model", "messages": []},
        )
    assert response.status_code == 400


def test_readiness_requires_lora_alias_to_be_loaded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base"}]})
        raise AssertionError(request.url.path)

    with TestClient(create_app(settings(), transport=httpx.MockTransport(handler))) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["reason"] == "required_model_missing"


def test_readiness_passes_when_required_model_is_loaded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "tool-calling"}]})
        raise AssertionError(request.url.path)

    with TestClient(create_app(settings(), transport=httpx.MockTransport(handler))) as client:
        response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_streaming_sse_is_proxied() -> None:
    payload = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    class StaticAsyncStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield payload

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=StaticAsyncStream(),
        )

    with (
        TestClient(create_app(settings(), transport=httpx.MockTransport(handler))) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"authorization": "Bearer public-a", "x-request-id": "req-123"},
            json={"model": "tool-calling", "messages": [], "stream": True},
        ) as response,
    ):
        content = b"".join(response.iter_bytes())
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"
    assert b"[DONE]" in content


def test_metrics_expose_low_cardinality_gateway_series() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": [{"id": "tool-calling"}]})
    )
    with TestClient(create_app(settings(), transport=transport)) as client:
        client.get("/v1/models", headers={"authorization": "Bearer public-a"})
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "qwen_gateway_requests_total" in response.text
    assert 'route="models"' in response.text


def test_request_body_limit_rejects_before_proxying() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    tiny = settings(max_request_bytes=10)
    with TestClient(create_app(tiny, transport=httpx.MockTransport(handler))) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer public-a"},
            content=b'{"model":"tool-calling"}',
        )
    assert response.status_code == 413
    assert calls == 0
