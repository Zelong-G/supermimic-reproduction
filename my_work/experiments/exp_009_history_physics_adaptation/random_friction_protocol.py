"""Pure, reproducible schedule and pairing rules for Exp009 (no simulator imports)."""

import hashlib

import numpy as np

DT = 0.02
SWITCH_STEPS = 150


def schedule(name, seed, steps, control=False):
    """Uniform[0.2,1.5], first 3s at 1.0, change before steps 150,300,...

    Stable per-motion RNG: changing clip selection/order cannot change a schedule.
    Schedule is independent of actor and survival; it never restarts on failure.
    """
    digest = hashlib.sha256(name.encode()).digest()
    rng = np.random.default_rng(np.random.SeedSequence([seed, *np.frombuffer(digest, dtype="<u4")]))
    values = np.ones(steps, dtype=np.float32)
    if not control:
        for start in range(SWITCH_STEPS, steps, SWITCH_STEPS):
            values[start : start + SWITCH_STEPS] = rng.uniform(0.2, 1.5)
    return values


def prefix_audit(control, changed, atol=1e-4):
    """Within-policy no-switch vs random-switch gate, never cross-policy equality."""
    if control.shape != changed.shape or not len(control):
        return {
            "passed": False,
            "max_abs": None,
            "atol": atol,
            "reason": "Missing or unequal prefix",
        }
    diff = float(np.max(np.abs(control.astype(float) - changed.astype(float))))
    return {"passed": bool(np.isfinite(diff) and diff <= atol), "max_abs": diff, "atol": atol}


def clustered_interval(values, seed=9):
    """Input: one value per independent parent motion, seeds already averaged."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None, None
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), (2000, len(x)))].mean(-1)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]
