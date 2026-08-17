from __future__ import annotations

import argparse
import asyncio
import json
from argparse import Namespace
from pathlib import Path

from scripts.capture_runtime import collect
from scripts.evaluate_tool_calling import run as run_tool_eval
from scripts.sweep import run as run_sweep


async def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = collect(args.base_url, args.api_key)
    (output_dir / "runtime_metadata.json").write_text(json.dumps(runtime, indent=2) + "\n")

    tool_args = Namespace(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        dataset=args.tool_dataset,
        tools=args.tools,
        policy=args.tool_policy,
        concurrency=args.tool_eval_concurrency,
        max_tokens=args.tool_eval_max_tokens,
        timeout=args.timeout,
    )
    tool_report = await run_tool_eval(tool_args)
    (output_dir / "tool_calling.json").write_text(json.dumps(tool_report, indent=2) + "\n")

    sweep_args = Namespace(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        concurrencies=args.concurrencies,
        requests_per_point=args.requests_per_point,
        max_tokens=args.benchmark_max_tokens,
        timeout=args.timeout,
        slo_policy=args.slo_policy,
        prompt=args.prompt,
    )
    sweep_report = await run_sweep(sweep_args)
    (output_dir / "concurrency_sweep.json").write_text(json.dumps(sweep_report, indent=2) + "\n")

    summary = {
        "runtime": runtime,
        "tool_calling": tool_report["summary"],
        "tool_policy": tool_report.get("policy"),
        "best_slo_point": sweep_report.get("best_slo_point"),
    }
    (output_dir / "suite_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete measured GPU serving suite.")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="change-me-public")
    parser.add_argument("--model", default="tool-calling")
    parser.add_argument("--tool-dataset", default="evals/tool_calling_v1.jsonl")
    parser.add_argument("--tools", default="evals/tools_v1.json")
    parser.add_argument("--tool-policy", default="evals/tool_calling_targets.json")
    parser.add_argument("--slo-policy", default="evals/slo_policy.json")
    parser.add_argument("--tool-eval-concurrency", type=int, default=1)
    parser.add_argument("--tool-eval-max-tokens", type=int, default=256)
    parser.add_argument("--concurrencies", default="1,2,4,8")
    parser.add_argument("--requests-per-point", type=int, default=32)
    parser.add_argument("--benchmark-max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--prompt",
        default="Reply with one short sentence about reliable model serving.",
    )
    parser.add_argument("--output-dir", default="evals/results/gpu_suite")
    return parser.parse_args()


def main() -> None:
    summary = asyncio.run(run(parse_args()))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
