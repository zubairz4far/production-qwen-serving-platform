from __future__ import annotations

import argparse
import asyncio
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from scripts.benchmark import run as run_benchmark
from scripts.capture_runtime import collect
from scripts.evaluate_slo import evaluate


def parse_concurrencies(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise ValueError("Concurrencies must be positive integers.")
    return values


def select_best(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [point for point in points if point.get("slo_pass")]
    if not passing:
        return None
    return max(passing, key=lambda point: point["throughput_requests_per_second"])


async def run(args: argparse.Namespace) -> dict[str, Any]:
    policy = json.loads(Path(args.slo_policy).read_text())
    points: list[dict[str, Any]] = []
    for concurrency in parse_concurrencies(args.concurrencies):
        benchmark_args = Namespace(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            requests=args.requests_per_point,
            concurrency=concurrency,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            prompt=args.prompt,
        )
        point = await run_benchmark(benchmark_args)
        failures = evaluate(point, policy)
        point["slo_pass"] = not failures
        point["slo_failed_checks"] = failures
        points.append(point)

    return {
        "runtime": collect(args.base_url, args.api_key),
        "model": args.model,
        "requests_per_point": args.requests_per_point,
        "max_tokens": args.max_tokens,
        "points": points,
        "best_slo_point": select_best(points),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep serving concurrency against SLO targets.")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="change-me-public")
    parser.add_argument("--model", default="tool-calling")
    parser.add_argument("--concurrencies", default="1,2,4,8")
    parser.add_argument("--requests-per-point", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--slo-policy", default="evals/slo_policy.json")
    parser.add_argument(
        "--prompt",
        default="Reply with one short sentence about reliable model serving.",
    )
    parser.add_argument("--output", default="evals/results/concurrency_sweep.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
