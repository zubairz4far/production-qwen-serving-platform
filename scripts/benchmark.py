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


async def one_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    async with semaphore:
        started = time.perf_counter()
        ttft_ms: float | None = None
        output_bytes = 0
        status_code = 0
        error: str | None = None
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
                },
            ) as response:
                status_code = response.status_code
                async for chunk in response.aiter_bytes():
                    if chunk and ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    output_bytes += len(chunk)
        except httpx.HTTPError as exc:
            error = type(exc).__name__
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "status_code": status_code,
            "ok": 200 <= status_code < 300 and error is None,
            "ttft_ms": ttft_ms,
            "total_latency_ms": total_ms,
            "output_bytes": output_bytes,
            "error": error,
        }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    headers = {"authorization": f"Bearer {args.api_key}"}
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    timeout = httpx.Timeout(args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    wall_started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), headers=headers, limits=limits, timeout=timeout
    ) as client:
        results = await asyncio.gather(
            *[
                one_request(client, semaphore, args.model, args.prompt, args.max_tokens)
                for _ in range(args.requests)
            ]
        )
    wall_seconds = time.perf_counter() - wall_started
    successful = [item for item in results if item["ok"]]
    totals = [item["total_latency_ms"] for item in successful]
    ttfts = [item["ttft_ms"] for item in successful if item["ttft_ms"] is not None]
    return {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "successful_requests": len(successful),
        "failed_requests": args.requests - len(successful),
        "success_rate": len(successful) / args.requests if args.requests else 0.0,
        "wall_seconds": wall_seconds,
        "throughput_requests_per_second": len(successful) / wall_seconds if wall_seconds else 0.0,
        "latency_ms": {
            "mean": statistics.fmean(totals) if totals else 0.0,
            "p50": percentile(totals, 0.50),
            "p95": percentile(totals, 0.95),
            "p99": percentile(totals, 0.99),
        },
        "ttft_ms": {
            "mean": statistics.fmean(ttfts) if ttfts else 0.0,
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
        },
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
    parser.add_argument("--prompt", default="Reply with one short sentence about reliable model serving.")
    parser.add_argument("--output", default="evals/results/latest_benchmark.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summary = asyncio.run(run(arguments))
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
