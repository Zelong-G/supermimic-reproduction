"""Pairing/quality tests; synthetic rows are not experiment measurements."""

import pytest
from sweep_report import paired_effect


def row(name, value, cap=0):
    return {
        "name": name,
        "norm_return": value,
        "capacity_overflow_steps": cap,
        "initial_state_sha256": name,
    }


def test_control_floor_and_clean_intersection():
    b = [row("a", 0.8), row("b", 0.5)]
    a = [row("a", 0.9), row("b", 0.7, 1)]
    d = [row("a", 0.81), row("b", 0.55)]
    s = paired_effect(b, a, d, clean=True)
    assert s["n"] == 1 and abs(s["delta"] - 0.1) < 1e-12
    assert abs(s["disabled_absolute_delta"] - 0.01) < 1e-12
    assert paired_effect(b, a, d)["n"] == 2


def test_initial_state_mismatch_fails():
    b = [row("a", 0.8)]
    a = [row("a", 0.9)]
    a[0]["initial_state_sha256"] = "different"
    with pytest.raises(ValueError):
        paired_effect(b, a)


def test_unavailable_clean_subset_stays_null():
    s = paired_effect([row("a", 0.8)], [row("a", 0.9, 1)], clean=True)
    assert s["n"] == 0 and s["delta"] is None and s["low"] is None
