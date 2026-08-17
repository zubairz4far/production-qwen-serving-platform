import pytest

from scripts.sweep import parse_concurrencies, select_best


def test_parse_concurrencies() -> None:
    assert parse_concurrencies("1,2, 4,8") == [1, 2, 4, 8]
    with pytest.raises(ValueError):
        parse_concurrencies("0,2")


def test_select_best_uses_highest_throughput_passing_point() -> None:
    points = [
        {"concurrency": 1, "slo_pass": True, "throughput_requests_per_second": 2.0},
        {"concurrency": 2, "slo_pass": False, "throughput_requests_per_second": 7.0},
        {"concurrency": 4, "slo_pass": True, "throughput_requests_per_second": 5.0},
    ]
    assert select_best(points)["concurrency"] == 4
