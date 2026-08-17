from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class GatewayMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "qwen_gateway_requests_total",
            "Requests handled by the Qwen gateway.",
            ["route", "status_class"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "qwen_gateway_request_duration_seconds",
            "Gateway request duration before streaming body consumption.",
            ["route"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            registry=self.registry,
        )
        self.inflight = Gauge(
            "qwen_gateway_inflight_requests",
            "Requests currently holding a generation concurrency slot.",
            registry=self.registry,
        )
        self.queue_rejections = Counter(
            "qwen_gateway_queue_rejections_total",
            "Requests rejected because the generation concurrency queue timed out.",
            registry=self.registry,
        )
        self.upstream_errors = Counter(
            "qwen_gateway_upstream_errors_total",
            "Transport failures talking to the vLLM upstream.",
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
