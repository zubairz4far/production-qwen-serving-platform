from scripts.benchmark import distribution, parse_sse_data


def test_parse_sse_usage_metrics_chunk() -> None:
    payload = parse_sse_data(
        'data: {"choices":[],"usage":{"completion_tokens":8},'
        '"metrics":{"time_to_first_token_ms":42.5}}'
    )
    assert payload is not None
    assert payload["usage"]["completion_tokens"] == 8
    assert payload["metrics"]["time_to_first_token_ms"] == 42.5


def test_parse_sse_ignores_done_and_invalid_json() -> None:
    assert parse_sse_data("data: [DONE]") is None
    assert parse_sse_data("data: {not-json") is None
    assert parse_sse_data("event: message") is None


def test_distribution_is_deterministic() -> None:
    summary = distribution([10.0, 20.0, 30.0, 40.0])
    assert summary == {"mean": 25.0, "p50": 20.0, "p95": 40.0, "p99": 40.0}
