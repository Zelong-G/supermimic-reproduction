"""Coordinate-consistent diagnostics of benchmark Recording arrays (no simulation)."""

from __future__ import annotations

import numpy as np


def control_timing(physics_dt, decimation, expected_fps=50):
    """Use the environment config; exported MjSpec options may be unapplied defaults."""
    if not np.isfinite(physics_dt) or physics_dt <= 0 or decimation < 1:
        raise ValueError("Invalid environment timing")
    dt = float(physics_dt * decimation)
    if not np.isclose(dt, 1 / expected_fps):
        raise ValueError(f"Configured control timestep {dt} differs from {1 / expected_fps}")
    return {
        "physics_dt_s": float(physics_dt),
        "decimation": int(decimation),
        "control_dt_s": dt,
        "source": "community environment configuration",
    }


def root_errors(root, reference):
    """Rows [x,y,z,qw,qx,qy,qz] in the SAME origin-free world coordinates."""
    root, reference = np.asarray(root, dtype=float), np.asarray(reference, dtype=float)
    if root.shape != reference.shape or root.ndim != 2 or root.shape[1] != 7:
        raise ValueError("Expected matching (N, 7) root pose arrays")
    if not np.isfinite(root).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite root poses")
    q, r = root[:, 3:].copy(), reference[:, 3:].copy()
    qnorm, rnorm = np.linalg.norm(q, axis=1), np.linalg.norm(r, axis=1)
    if np.any(qnorm < 1e-8) or np.any(rnorm < 1e-8):
        raise ValueError("Zero quaternion")
    q /= qnorm[:, None]
    r /= rnorm[:, None]
    angle = 2 * np.arccos(np.clip(np.abs(np.sum(q * r, axis=1)), 0, 1))
    # z component of inverse-rotated unit gravity; sign cancels in abs difference.
    tilt = np.abs((1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)) - (1 - 2 * (r[:, 1] ** 2 + r[:, 2] ** 2)))
    return {
        "position_error_m": np.linalg.norm(root[:, :3] - reference[:, :3], axis=1),
        "orientation_error_rad": angle,
        "height_error_m": np.abs(root[:, 2] - reference[:, 2]),
        "benchmark_tilt_error": tilt,
    }


def compare_outcome(historical, failed, fail_step, dt=0.02):
    """An unmeasured result must never be reported as a match."""
    if failed is None:
        return {
            "status": "not_measured",
            "historical_failed": historical["failed"],
            "historical_fail_step": historical["fail_step"],
        }
    same = bool(failed) == historical["failed"]
    delta = int(fail_step) - historical["fail_step"] if failed and historical["failed"] else None
    return {
        "status": "exact_outcome_match" if same and (not failed or delta == 0) else "mismatch",
        "historical_failed": historical["failed"],
        "historical_fail_step": historical["fail_step"],
        "observed_failed": bool(failed),
        "observed_fail_step": int(fail_step),
        "fail_step_difference": delta,
        "fail_time_difference_s": None if delta is None else delta * dt,
    }


def video_window(length, fail_step, fps=50, seconds=12, start=None):
    """Exclusive stop; clip failures terminate before any post-failure footage exists."""
    if length < 1 or seconds <= 0:
        raise ValueError("Empty recording or nonpositive video duration")
    count = max(1, round(seconds * fps))
    begin = max(0, (min(length, fail_step) - count) if fail_step >= 0 else 0)
    if start is not None:
        if not np.isfinite(start) or start < 0 or start * fps >= length:
            raise ValueError("Video start is outside the recorded trajectory")
        begin = round(start * fps)
        begin = min(begin, length - 1)
    return begin, min(length, begin + count)
