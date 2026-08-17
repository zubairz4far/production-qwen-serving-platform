from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(result: dict, policy: dict) -> list[str]:
    failures: list[str] = []
    if result["success_rate"] < policy["min_success_rate"]:
        failures.append("success_rate")
    if result["latency_ms"]["p95"] > policy["max_p95_latency_ms"]:
        failures.append("p95_latency_ms")
    if result["ttft_ms"]["p95"] > policy["max_p95_ttft_ms"]:
        failures.append("p95_ttft_ms")
    if result["throughput_requests_per_second"] < policy["min_requests_per_second"]:
        failures.append("requests_per_second")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--policy", default="evals/slo_policy.json")
    args = parser.parse_args()

    result = json.loads(Path(args.result).read_text())
    policy = json.loads(Path(args.policy).read_text())
    failures = evaluate(result, policy)
    print(json.dumps({"passed": not failures, "failed_checks": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
