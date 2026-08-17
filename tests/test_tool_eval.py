from __future__ import annotations

import json
from pathlib import Path

from scripts.tool_eval_core import evaluate_policy, parse_tool_calls, score_case, summarize


def tool_payload(name: str, arguments: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            }
        ]
    }


def test_parses_openai_tool_call_shape() -> None:
    calls, valid, reason = parse_tool_calls(tool_payload("get_order", {"order_id": "123"}))
    assert valid is True
    assert reason == "ok"
    assert calls == [{"name": "get_order", "arguments": {"order_id": "123"}}]


def test_rejects_malformed_tool_arguments() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "get_order", "arguments": "{not-json"}}
                    ]
                }
            }
        ]
    }
    calls, valid, reason = parse_tool_calls(payload)
    assert calls == []
    assert valid is False
    assert reason == "arguments_not_json"


def test_tool_case_requires_exact_tool_and_arguments() -> None:
    case = {
        "id": "x",
        "expected_behavior": "tool",
        "expected_tool": "get_weather",
        "expected_args": {"city": "Lahore"},
        "category": "tool_required",
    }
    scored = score_case(case, tool_payload("get_weather", {"city": "Lahore"}))
    assert scored["passed"] is True
    wrong = score_case(case, tool_payload("get_weather", {"city": "Karachi"}))
    assert wrong["passed"] is False


def test_clarification_requires_no_tool_and_nonempty_text() -> None:
    case = {
        "id": "clarify",
        "expected_behavior": "clarify",
        "expected_tool": "get_order",
        "expected_args": {},
        "category": "missing_order_id",
    }
    payload = {"choices": [{"message": {"content": "What is the order ID?", "tool_calls": []}}]}
    assert score_case(case, payload)["passed"] is True
    assert score_case(case, tool_payload("get_order", {"order_id": "invented"}))["passed"] is False


def test_no_tool_case_rejects_hallucinated_call() -> None:
    case = {
        "id": "no-tool",
        "expected_behavior": "no_tool",
        "expected_tool": None,
        "expected_args": {},
        "category": "hard_negative",
    }
    payload = {"choices": [{"message": {"content": "Explanation", "tool_calls": []}}]}
    assert score_case(case, payload)["passed"] is True
    assert score_case(case, tool_payload("get_order", {"order_id": "999"}))["passed"] is False


def test_summary_and_policy_fixture_are_deterministic() -> None:
    scored = [
        {"category":"tool_required","expected_behavior":"tool","passed":True,"parse_valid":True,"tool_call_count":1,"tool_selection_correct":True,"strict_call_correct":True},
        {"category":"prompt_injection","expected_behavior":"tool","passed":True,"parse_valid":True,"tool_call_count":1,"tool_selection_correct":True,"strict_call_correct":True},
        {"category":"missing_order_id","expected_behavior":"clarify","passed":True,"parse_valid":True,"tool_call_count":0,"tool_selection_correct":False,"strict_call_correct":False},
        {"category":"no_tool","expected_behavior":"no_tool","passed":True,"parse_valid":True,"tool_call_count":0,"tool_selection_correct":False,"strict_call_correct":False}
    ]
    summary = summarize(scored, [100.0, 120.0, 90.0, 80.0])
    assert summary["overall_accuracy"] == 1.0
    assert summary["strict_exact_call_accuracy"] == 1.0
    assert summary["hallucinated_tool_rate_on_non_tool_cases"] == 0.0
    policy = json.loads(Path("evals/tool_calling_targets.json").read_text())
    assert evaluate_policy(summary, policy) == []
