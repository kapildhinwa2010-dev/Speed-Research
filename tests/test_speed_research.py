"""Tests for Speed-Research: scaling tables and complexity exponents."""

import pytest

from speed_research import complexity_exponent, scaling_table, time_method


def test_time_method_positive():
    seconds = time_method(lambda: sum(range(1000)), repeats=2)
    assert seconds >= 0.0
    with pytest.raises(ValueError):
        time_method(lambda: None, repeats=0)


def test_scaling_table_covers_all_methods_and_sizes():
    rows = scaling_table(sizes=(16, 32), repeats=1, seed=0)
    assert {r["method"] for r in rows} == {"svd_rank4", "quantize8", "fourier4"}
    assert {r["size"] for r in rows} == {16, 32}


def test_complexity_exponent_linear_for_known_data():
    rows = [
        {"method": "m", "size": 10, "seconds": 1.0},
        {"method": "m", "size": 100, "seconds": 100.0},
    ]
    assert complexity_exponent(rows, "m") == pytest.approx(2.0, abs=1e-6)


def test_complexity_exponent_needs_two_sizes():
    with pytest.raises(ValueError):
        complexity_exponent([{"method": "m", "size": 8, "seconds": 0.1}], "m")


def test_scaling_table_validation():
    with pytest.raises(ValueError):
        scaling_table(sizes=(1,))
