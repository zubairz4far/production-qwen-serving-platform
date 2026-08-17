from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
from fastapi import HTTPException, status
from fastapi.responses import Response, StreamingResponse

from gateway.config import Settings
from gateway.metrics import GatewayMetrics

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class QueueTimeoutError(RuntimeError):
    pass


class UpstreamUnavailableError(RuntimeError):
    pass


class UpstreamProxy:
    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: Settings,
        metrics: GatewayMetrics,
    ) -> None:
        self.client = client
        self.settings = settings
        self.metrics = metrics
        self.semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def _acquire(self) -> None:
        try:
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.settings.queue_timeout_seconds,
            )
        except TimeoutError as exc:
            self.metrics.queue_rejections.inc()
            raise QueueTimeoutError from exc
        self.metrics.inflight.inc()

    def _release(self) -> None:
        self.metrics.inflight.dec()
        self.semaphore.release()

    def upstream_headers(self, request_id: str) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.settings.upstream_api_key}",
            "content-type": "application/json",
            "x-request-id": request_id,
        }

    @staticmethod
    def _response_headers(response: httpx.Response, request_id: str) -> dict[str, str]:
        headers = {"x-request-id": request_id}
        for name, value in response.headers.items():
            lowered = name.lower()
            if lowered in HOP_BY_HOP_HEADERS or lowered == "content-length":
                continue
            if lowered in {"content-type", "cache-control"}:
                headers[lowered] = value
        return headers

    async def get(self, path: str, request_id: str) -> Response:
        try:
            response = await self.client.get(
                path,
                headers=self.upstream_headers(request_id),
            )
        except httpx.HTTPError as exc:
            self.metrics.upstream_errors.inc()
            raise UpstreamUnavailableError from exc
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._response_headers(response, request_id),
            media_type=None,
        )

    async def post_json(
        self,
        path: str,
        body: dict,
        request_id: str,
        *,
        stream: bool,
    ) -> Response:
        await self._acquire()
        request = self.client.build_request(
            "POST",
            path,
            json=body,
            headers=self.upstream_headers(request_id),
        )
        try:
            response = await self.client.send(request, stream=True)
        except httpx.HTTPError as exc:
            self._release()
            self.metrics.upstream_errors.inc()
            raise UpstreamUnavailableError from exc

        headers = self._response_headers(response, request_id)
        if not stream:
            try:
                content = await response.aread()
            finally:
                await response.aclose()
                self._release()
            return Response(
                content=content,
                status_code=response.status_code,
                headers=headers,
                media_type=None,
            )

        async def iterator() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                self._release()

        return StreamingResponse(
            iterator(),
            status_code=response.status_code,
            headers=headers,
            media_type=None,
        )


def queue_timeout_http_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Inference capacity is saturated. Retry later.",
        headers={"Retry-After": "1"},
    )
