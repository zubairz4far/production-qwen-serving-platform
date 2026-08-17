from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def parse_sse_data(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


async def one_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    async with semaphore:
        started = time.perf_counter()
        client_ttft_ms: float | None = None
        output_bytes = 0
        status_code = 0
        error: str | None = None
        usage: dict[str, Any] = {}
        engine_metrics: dict[str, Any] = {}
        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            ) as response:
                status_code = response.status_code
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    output_bytes += len(line.encode("utf-8"))
                    payload = parse_sse_data(line)
                    if payload is None:
                        continue
                    if client_ttft_ms is None and payload.get("choices"):
                        client_ttft_ms = (time.perf_counter() - started) * 1000
                    if isinstance(payload.get("usage"), dict):
                        usage = payload["usage"]
                    if isinstance(payload.get("metrics"), dict):
                        engine_metrics = payload["metrics"]
        except httpx.HTTPError as exc:
            error = type(exc).__name__

        total_ms = (time.perf_counter() - started) * 1000
        return {
            "status_code": status_code,
            "ok": 200 <= status_code < 300 and error is None,
            "client_ttft_ms": client_ttft_ms,
            "total_latency_ms": total_ms,
            "output_bytes": output_bytes,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "engine_ttft_ms": engine_metrics.get("time_to_first_token_ms"),
            "engine_queue_time_ms": engine_metrics.get("queue_time_ms"),
            "engine_mean_itl_ms": engine_metrics.get("mean_itl_ms"),
            "engine_tokens_per_second": engine_metrics.get("tokens_per_second"),
            "server_metrics_available": bool(engine_metrics),
            "error": error,
        }


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    return values


async def run(args: argparse.Namespace) -> dict[str, Any]:
    headers = {"authorization": f"Bearer {args.api_key}"}
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    wall_started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        limits=limits,
        timeout=timeout,
    ) as client:
        results = await asyncio.gather(
            *[
                one_request(client, semaphore, args.model, args.prompt, args.max_tokens)
                for _ in range(args.requests)
            ]
        )

    wall_seconds = time.perf_counter() - wall_started
    successful = [item for item in results if item["ok"]]
    totals = numeric_values(successful, "total_latency_ms")
    client_ttfts = numeric_values(successful, "client_ttft_ms")
    engine_ttfts = numeric_values(successful, "engine_ttft_ms")
    queue_times = numeric_values(successful, "engine_queue_time_ms")
    itls = numeric_values(successful, "engine_mean_itl_ms")
    engine_tps = numeric_values(successful, "engine_tokens_per_second")
    completion_tokens = numeric_values(successful, "completion_tokens")
    total_completion_tokens = int(sum(completion_tokens))

    return {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "successful_requests": len(successful),
        "failed_requests": args.requests - len(successful),
        "success_rate": len(successful) / args.requests if args.requests else 0.0,
        "wall_seconds": wall_seconds,
        "throughput_requests_per_second": len(successful) / wall_seconds if wall_seconds else 0.0,
        "throughput_output_tokens_per_second": (
            total_completion_tokens / wall_seconds if wall_seconds else 0.0
        ),
        "completion_tokens": total_completion_tokens,
        "latency_ms": distribution(totals),
        "ttft_ms": distribution(client_ttfts),
        "engine_ttft_ms": distribution(engine_ttfts),
        "engine_queue_time_ms": distribution(queue_times),
        "engine_mean_itl_ms": distribution(itls),
        "engine_tokens_per_second": distribution(engine_tps),
        "server_metrics_coverage": len(engine_ttfts) / len(successful) if successful else 0.0,
        "failures": [item for item in results if not item["ok"]][:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="change-me-public")
    parser.add_argument("--model", default="tool-calling")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--prompt",
        default="Reply with one short sentence about reliable model serving.",
    )
    parser.add_argument("--output", default="evals/results/latest_benchmark.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summary = asyncio.run(run(arguments))
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
