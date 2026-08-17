from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response

from gateway.auth import require_api_key
from gateway.config import Settings, get_settings
from gateway.metrics import GatewayMetrics
from gateway.proxy import (
    QueueTimeoutError,
    UpstreamProxy,
    UpstreamUnavailableError,
    queue_timeout_http_error,
)


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid4())


def _status_class(code: int) -> str:
    return f"{code // 100}xx"


async def _read_bounded_body(request: Request, limit: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > limit:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Request body too large.",
                )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from None

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Request body too large.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    resolved = settings or get_settings()
    metrics = GatewayMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = httpx.Timeout(resolved.upstream_timeout_seconds, connect=10.0)
        async with httpx.AsyncClient(
            base_url=resolved.upstream_base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        ) as client:
            app.state.proxy = UpstreamProxy(client, resolved, metrics)
            yield

    app = FastAPI(
        title=resolved.app_name,
        version="0.1.0",
        description=(
            "Authenticated, backpressure-aware OpenAI-compatible edge gateway for a vLLM-backed "
            "Qwen serving deployment."
        ),
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.metrics = metrics

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = _request_id(request)
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready(request: Request) -> JSONResponse:
        request_id = request.state.request_id
        proxy: UpstreamProxy = request.app.state.proxy
        try:
            health = await proxy.client.get("/health", timeout=3.0)
            models = await proxy.client.get(
                "/v1/models",
                headers=proxy.upstream_headers(request_id),
                timeout=3.0,
            )
        except httpx.HTTPError:
            return JSONResponse(status_code=503, content={"status": "not_ready"})

        if health.status_code >= 400 or models.status_code >= 400:
            return JSONResponse(status_code=503, content={"status": "not_ready"})

        try:
            model_ids = {item["id"] for item in models.json().get("data", [])}
        except (ValueError, KeyError, TypeError):
            return JSONResponse(status_code=503, content={"status": "not_ready"})

        if resolved.readiness_model and resolved.readiness_model not in model_ids:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "required_model_missing"},
            )
        return JSONResponse(content={"status": "ready"})

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        require_api_key(request, resolved)
        proxy: UpstreamProxy = request.app.state.proxy
        request_id = request.state.request_id
        started = time.perf_counter()
        try:
            response = await proxy.get("/v1/models", request_id)
        except UpstreamUnavailableError as exc:
            raise HTTPException(status_code=502, detail="Inference upstream unavailable.") from exc
        finally:
            metrics.request_latency.labels(route="models").observe(time.perf_counter() - started)
        metrics.requests.labels(
            route="models",
            status_class=_status_class(response.status_code),
        ).inc()
        return response

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        require_api_key(request, resolved)
        raw = await _read_bounded_body(request, resolved.max_request_bytes)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object.")

        model = body.get("model")
        if not isinstance(model, str) or model not in resolved.allowed_model_set:
            raise HTTPException(status_code=400, detail="Model is not allowed by this gateway.")

        stream = body.get("stream") is True
        proxy: UpstreamProxy = request.app.state.proxy
        request_id = request.state.request_id
        started = time.perf_counter()
        try:
            response = await proxy.post_json(
                "/v1/chat/completions",
                body,
                request_id,
                stream=stream,
            )
        except QueueTimeoutError as exc:
            raise queue_timeout_http_error() from exc
        except UpstreamUnavailableError as exc:
            raise HTTPException(status_code=502, detail="Inference upstream unavailable.") from exc
        finally:
            metrics.request_latency.labels(route="chat_completions").observe(
                time.perf_counter() - started
            )

        metrics.requests.labels(
            route="chat_completions",
            status_class=_status_class(response.status_code),
        ).inc()
        return response

    return app


app = create_app()
