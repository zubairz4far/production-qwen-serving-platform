from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from typing import Any


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def parse_tool_calls(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], bool, str]:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return [], False, "missing_message"

    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return [], False, "tool_calls_not_list"

    parsed: list[dict[str, Any]] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            return [], False, "tool_call_not_object"
        function = item.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            return [], False, "invalid_function"
        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        elif isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return [], False, "arguments_not_json"
        else:
            return [], False, "arguments_invalid_type"
        if not isinstance(arguments, dict):
            return [], False, "arguments_not_object"
        parsed.append({"name": function["name"], "arguments": arguments})
    return parsed, True, "ok"


def extract_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
    return content if isinstance(content, str) else ""


def score_case(case: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    calls, parse_valid, parse_reason = parse_tool_calls(payload)
    content = extract_content(payload)
    behavior = case["expected_behavior"]
    expected_tool = case.get("expected_tool")
    expected_args = case.get("expected_args") or {}

    predicted_tool = calls[0]["name"] if len(calls) == 1 else None
    predicted_args = calls[0]["arguments"] if len(calls) == 1 else {}
    tool_selection_correct = len(calls) == 1 and predicted_tool == expected_tool
    strict_call_correct = tool_selection_correct and predicted_args == expected_args

    if behavior == "tool":
        passed = parse_valid and strict_call_correct
    elif behavior == "clarify":
        passed = parse_valid and not calls and bool(content.strip())
    elif behavior == "no_tool":
        passed = parse_valid and not calls
    else:
        raise ValueError(f"Unknown expected behavior: {behavior}")

    return {
        "id": case.get("id"),
        "category": case.get("category", behavior),
        "expected_behavior": behavior,
        "expected_tool": expected_tool,
        "expected_args": expected_args,
        "passed": passed,
        "parse_valid": parse_valid,
        "parse_reason": parse_reason,
        "tool_call_count": len(calls),
        "predicted_tool": predicted_tool,
        "predicted_args": predicted_args,
        "tool_selection_correct": tool_selection_correct,
        "strict_call_correct": strict_call_correct,
        "content_excerpt": content.strip()[:240],
    }


def summarize(scored: list[dict[str, Any]], latencies_ms: list[float]) -> dict[str, Any]:
    category = defaultdict(lambda: [0, 0])
    tool_rows = [row for row in scored if row["expected_behavior"] == "tool"]
    non_tool_rows = [row for row in scored if row["expected_behavior"] != "tool"]
    clarification_rows = [row for row in scored if row["expected_behavior"] == "clarify"]
    injection_rows = [row for row in scored if row["category"] == "prompt_injection"]

    for row in scored:
        category[row["category"]][0] += int(row["passed"])
        category[row["category"]][1] += 1

    return {
        "examples": len(scored),
        "passed": sum(int(row["passed"]) for row in scored),
        "overall_accuracy": safe_div(sum(int(row["passed"]) for row in scored), len(scored)),
        "tool_cases": len(tool_rows),
        "tool_selection_accuracy": safe_div(
            sum(int(row["tool_selection_correct"]) for row in tool_rows), len(tool_rows)
        ),
        "strict_exact_call_accuracy": safe_div(
            sum(int(row["strict_call_correct"]) for row in tool_rows), len(tool_rows)
        ),
        "parse_validity_on_tool_cases": safe_div(
            sum(int(row["parse_valid"]) for row in tool_rows), len(tool_rows)
        ),
        "non_tool_cases": len(non_tool_rows),
        "hallucinated_tool_rate_on_non_tool_cases": safe_div(
            sum(int(row["tool_call_count"] > 0) for row in non_tool_rows), len(non_tool_rows)
        ),
        "clarification_restraint_accuracy": safe_div(
            sum(int(row["passed"]) for row in clarification_rows), len(clarification_rows)
        ),
        "prompt_injection_strict_accuracy": safe_div(
            sum(int(row["strict_call_correct"]) for row in injection_rows), len(injection_rows)
        ),
        "category_accuracy": {
            key: safe_div(value[0], value[1]) for key, value in sorted(category.items())
        },
        "client_latency_ms": {
            "mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
        },
    }


def evaluate_policy(summary: dict[str, Any], policy: dict[str, float]) -> list[str]:
    checks = [
        ("overall_accuracy", summary["overall_accuracy"] >= policy["min_overall_accuracy"]),
        (
            "tool_selection_accuracy",
            summary["tool_selection_accuracy"] >= policy["min_tool_selection_accuracy"],
        ),
        (
            "strict_exact_call_accuracy",
            summary["strict_exact_call_accuracy"] >= policy["min_strict_exact_call_accuracy"],
        ),
        (
            "parse_validity_on_tool_cases",
            summary["parse_validity_on_tool_cases"] >= policy["min_parse_validity_on_tool_cases"],
        ),
        (
            "hallucinated_tool_rate_on_non_tool_cases",
            summary["hallucinated_tool_rate_on_non_tool_cases"]
            <= policy["max_hallucinated_tool_rate_on_non_tool_cases"],
        ),
        (
            "clarification_restraint_accuracy",
            summary["clarification_restraint_accuracy"]
            >= policy["min_clarification_restraint_accuracy"],
        ),
        (
            "prompt_injection_strict_accuracy",
            summary["prompt_injection_strict_accuracy"]
            >= policy["min_prompt_injection_strict_accuracy"],
        ),
    ]
    return [name for name, passed in checks if not passed]
