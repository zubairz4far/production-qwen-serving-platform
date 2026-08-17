import json
from pathlib import Path


VALID_BEHAVIORS = {"tool", "clarify", "no_tool"}


def main() -> None:
    dataset_path = Path("evals/tool_calling_v1.jsonl")
    tools_path = Path("evals/tools_v1.json")
    cases = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    tools = json.loads(tools_path.read_text())

    ids = [case.get("id") for case in cases]
    if not cases or None in ids or len(ids) != len(set(ids)):
        raise SystemExit("Tool-calling cases must have unique non-empty IDs.")

    tool_names = {
        item["function"]["name"]
        for item in tools
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    if tool_names != {"get_order", "cancel_order", "get_weather"}:
        raise SystemExit(f"Unexpected tool set: {sorted(tool_names)}")

    counts: dict[str, int] = {}
    for case in cases:
        behavior = case.get("expected_behavior")
        if behavior not in VALID_BEHAVIORS:
            raise SystemExit(f"Invalid behavior in {case.get('id')}: {behavior}")
        if behavior == "tool" and case.get("expected_tool") not in tool_names:
            raise SystemExit(f"Invalid expected tool in {case.get('id')}")
        category = case.get("category", behavior)
        counts[category] = counts.get(category, 0) + 1

    required_categories = {
        "tool_required",
        "missing_order_id",
        "missing_city",
        "no_tool",
        "hard_negative",
        "prompt_injection",
    }
    missing = required_categories - set(counts)
    if missing:
        raise SystemExit(f"Missing evaluation categories: {sorted(missing)}")

    summary = {"cases": len(cases), "tools": sorted(tool_names), "categories": counts}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
