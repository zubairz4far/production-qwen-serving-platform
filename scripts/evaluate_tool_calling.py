from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from scripts.tool_eval_core import evaluate_policy, score_case, summarize


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def evaluate_case(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    case: dict[str, Any],
    tools: list[dict[str, Any]],
    model: str,
    max_tokens: int,
) -> tuple[dict[str, Any], float]:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": case["prompt"]}],
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            latency_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            scored = score_case(case, response.json())
            scored["http_status"] = response.status_code
            return scored, latency_ms
        except (httpx.HTTPError, ValueError) as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            failed = {
                "id": case.get("id"),
                "category": case.get("category"),
                "expected_behavior": case.get("expected_behavior"),
                "expected_tool": case.get("expected_tool"),
                "expected_args": case.get("expected_args") or {},
                "passed": False,
                "parse_valid": False,
                "parse_reason": type(exc).__name__,
                "tool_call_count": 0,
                "predicted_tool": None,
                "predicted_args": {},
                "tool_selection_correct": False,
                "strict_call_correct": False,
                "content_excerpt": "",
                "http_status": getattr(getattr(exc, "response", None), "status_code", 0),
            }
            return failed, latency_ms


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_jsonl(Path(args.dataset))
    tools = json.loads(Path(args.tools).read_text())
    if not cases:
        raise ValueError("Tool-calling dataset is empty.")
    if not isinstance(tools, list) or not tools:
        raise ValueError("Tool definition file must contain a non-empty JSON list.")

    headers = {"authorization": f"Bearer {args.api_key}"}
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        timeout=timeout,
        limits=limits,
    ) as client:
        results = await asyncio.gather(
            *[
                evaluate_case(client, semaphore, case, tools, args.model, args.max_tokens)
                for case in cases
            ]
        )

    scored = [item[0] for item in results]
    latencies = [item[1] for item in results]
    summary = summarize(scored, latencies)
    report: dict[str, Any] = {
        "dataset": args.dataset,
        "tools": args.tools,
        "model": args.model,
        "summary": summary,
        "cases": scored,
    }
    if args.policy:
        policy = json.loads(Path(args.policy).read_text())
        failures = evaluate_policy(summary, policy)
        report["policy"] = {
            "path": args.policy,
            "passed": not failures,
            "failed_checks": failures,
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate served OpenAI-compatible tool calling.")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="change-me-public")
    parser.add_argument("--model", default="tool-calling")
    parser.add_argument("--dataset", default="evals/tool_calling_v1.jsonl")
    parser.add_argument("--tools", default="evals/tools_v1.json")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", default="evals/results/tool_calling_latest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    policy = report.get("policy")
    if policy and not policy["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
