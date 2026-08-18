from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

PolicyAction = Literal["pass", "force_no_tool", "clarify"]


@dataclass(frozen=True)
class ToolPolicyDecision:
    action: PolicyAction
    body: dict[str, Any]
    tool_name: str | None = None
    clarification: str | None = None
    reason: str | None = None


_NON_EXECUTION_PATTERNS = [
    r"\bdo not (?:call|execute|run|perform|use)\b",
    r"\bdon't (?:call|execute|run|perform|use)\b",
    r"\bwithout (?:actually )?(?:calling|executing|running|performing|using)\b",
    r"\bno actual call\b",
    r"\bno real (?:tool )?call\b",
    r"\bno execution\b",
    r"\bno tool execution\b",
    r"\bdocumentation\b",
    r"\bmock example\b",
    r"\bsample (?:json|request|tool|call|input)\b",
    r"\bexample (?:json|request|tool|call|input)\b",
    r"\bquoted (?:text|documentation|instruction|command|tutorial)\b",
    r"\bexplain (?:this|the|how|what)\b.*\b(?:tool|call|request|command|instruction)\b",
    r"\bdescribe how\b.*\b(?:tool|request|call|order|weather)\b",
    r"\bfor (?:a )?(?:tutorial|example|documentation)\b",
    r"\bno real (?:order|cancellation|weather|supported|legitimate).*\brequest\b",
    r"\bno (?:actual|legitimate|supported) (?:user )?(?:action|request|task)\b",
]
_NON_EXECUTION_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in _NON_EXECUTION_PATTERNS),
    re.IGNORECASE,
)

_ORDER_CONTEXT_RE = re.compile(
    r"\border(?:\s*(?:id|number|#))?\s*[:#-]?\s*(\d{3,})\b",
    re.IGNORECASE,
)
_EXPLICIT_NUMBER_RE = re.compile(
    r"\b(?:with|using|use|value|id|number|#)\s*[:#-]?\s*(\d{3,})\b",
    re.IGNORECASE,
)
_ANY_LONG_NUMBER_RE = re.compile(r"\b(\d{4,})\b")

_CITY_PATTERNS = [
    re.compile(
        r"\bweather\s+(?:in|for|with|using)\s+"
        r"([A-Za-z][A-Za-z .'-]{1,40}?)(?:[?.!,;]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:with|using|use|value)\s+(?:the\s+explicit\s+value\s+)?"
        r"([A-Za-z][A-Za-z .'-]{1,40}?)(?:[?.!,;]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:in|for)\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?:[?.!,;]|$)",
        re.IGNORECASE,
    ),
]

_CITY_STOP_VALUES = {
    "me",
    "my",
    "my request",
    "the request",
    "your request",
    "this",
    "that",
    "there",
    "here",
    "it",
    "the weather",
    "current conditions",
    "the required value",
    "required value",
    "the explicit value",
}


def _message_text(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return None


def latest_user_text(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _message_text(message)
        if text is not None:
            return text
    return ""


def user_context_text(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _message_text(message)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _has_non_execution_intent(text: str) -> bool:
    return bool(_NON_EXECUTION_RE.search(text))


def _infer_tool(text: str) -> str | None:
    lowered = text.lower()

    if "cancel" in lowered and "order" in lowered:
        return "cancel_order"

    if "weather" in lowered:
        return "get_weather"

    order_intent_words = (
        "check",
        "status",
        "look up",
        "lookup",
        "retrieve",
        "access",
        "fetch",
        "find",
        "details",
        "current",
    )
    if "order" in lowered and any(word in lowered for word in order_intent_words):
        return "get_order"

    return None


def _extract_order_id(text: str) -> str | None:
    for pattern in (_ORDER_CONTEXT_RE, _EXPLICIT_NUMBER_RE, _ANY_LONG_NUMBER_RE):
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def _clean_city_candidate(value: str) -> str:
    cleaned = value.strip()
    for suffix in (" right now", " now"):
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    return cleaned


def _extract_city(text: str) -> str | None:
    for pattern in _CITY_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        candidate = _clean_city_candidate(match.group(1))
        if candidate.lower() in _CITY_STOP_VALUES:
            continue
        return candidate
    return None


def _extract_required_value(tool_name: str, context: str) -> str | None:
    if tool_name in {"get_order", "cancel_order"}:
        return _extract_order_id(context)
    if tool_name == "get_weather":
        return _extract_city(context)
    return None


def _clarification_for(tool_name: str) -> str:
    if tool_name in {"get_order", "cancel_order"}:
        return "What order ID should I use?"
    if tool_name == "get_weather":
        return "Which city should I check?"
    return "Could you provide the required value?"


def apply_tool_policy(body: dict[str, Any]) -> ToolPolicyDecision:
    latest = latest_user_text(body)
    if not latest:
        return ToolPolicyDecision(action="pass", body=body)

    if _has_non_execution_intent(latest):
        rewritten = dict(body)
        rewritten["tool_choice"] = "none"
        return ToolPolicyDecision(
            action="force_no_tool",
            body=rewritten,
            reason="explicit_non_execution_intent",
        )

    tool_name = _infer_tool(latest)
    if tool_name is None:
        return ToolPolicyDecision(action="pass", body=body)

    context = user_context_text(body)
    required_value = _extract_required_value(tool_name, context)
    if required_value is None:
        return ToolPolicyDecision(
            action="clarify",
            body=body,
            tool_name=tool_name,
            clarification=_clarification_for(tool_name),
            reason="missing_required_argument",
        )

    return ToolPolicyDecision(
        action="pass",
        body=body,
        tool_name=tool_name,
        reason="grounded_tool_request",
    )
