"""Statistical alignment and numerical-quality gates (synthetic unit fixtures only)."""

from results import paired, paired_summary


def fixture():
    rows = []
    for name in ["a", "b"]:
        for variant, value in [("base", 0.8), ("adapted", 0.9), ("residual_disabled", 0.801)]:
            rows.append(
                {
                    "name": name,
                    "variant": variant,
                    "mu": 0.3,
                    "norm_return": value,
                    "capacity_overflow_steps": int(name == "b" and variant == "adapted"),
                    "initial_state_sha256": name,
                }
            )
    return rows


def test_common_clean_intersection():
    rows = fixture()
    assert len(paired(rows, 0.3)) == 2
    clean = paired(rows, 0.3, clean=True)
    assert len(clean) == 1 and clean[0]["name"] == "a"


def test_paired_uncertainty_is_not_frame_resampling():
    s = paired_summary(paired(fixture(), 0.3))
    assert abs(s["mean_delta"] - 0.1) < 1e-12
    assert s["bootstrap_low"] == s["bootstrap_high"] == s["mean_delta"]
    assert abs(s["mean_abs_disabled_delta"] - 0.001) < 1e-12


def test_empty_clean_subset_is_unavailable():
    s = paired_summary([])
    assert s["episodes"] == 0 and s["mean_delta"] is None
