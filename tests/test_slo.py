from scripts.evaluate_slo import evaluate


def test_slo_evaluator_reports_specific_failures() -> None:
    policy = {
        "min_success_rate": 0.99,
        "max_p95_latency_ms": 5000,
        "max_p95_ttft_ms": 1500,
        "min_requests_per_second": 1.0,
    }
    result = {
        "success_rate": 0.95,
        "throughput_requests_per_second": 0.5,
        "latency_ms": {"p95": 6000},
        "ttft_ms": {"p95": 1000},
    }
    assert evaluate(result, policy) == ["success_rate", "p95_latency_ms", "requests_per_second"]
