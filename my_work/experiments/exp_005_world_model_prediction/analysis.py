"""Aligned inference and metrics; no physics or policy implementation."""

from __future__ import annotations

import numpy as np
import torch


def validate_trajectory(data):
    s, a, xi = data["states"], data["actions"], data["xi"]
    t = len(a)
    if t < 32 or s.shape != (t + 1, 71) or a.shape != (t, 29) or xi.shape != (t, 33):
        raise ValueError("Expected contiguous s[T+1,71], executed a[T,29], xi[T,33]; T>=32")
    for name in ["states", "actions", "xi"]:
        if not np.isfinite(data[name]).all():
            raise ValueError(f"Nonfinite trajectory: {name}")
    for name in ["terminated", "truncated", "overflow"]:
        if data[name].shape != (t,):
            raise ValueError(f"Invalid transition flags: {name}")
    if np.any(data["terminated"][:-1] | data["truncated"][:-1]):
        raise ValueError("Trajectory crosses a reset or done boundary")
    if not np.allclose(np.linalg.norm(s[:, 3:7], axis=1), 1, atol=1e-4):
        raise ValueError("Root quaternions must be normalized")
    if not np.isclose(float(data["dt"]), 0.02):
        raise ValueError("Trajectory/model timestep mismatch")


def window_starts(length, horizon=32, stride=8, maximum=64):
    if length < horizon or stride < 1 or maximum < 1:
        raise ValueError("No valid windows or invalid sampling configuration")
    starts = np.arange(0, length - horizon + 1, stride)
    if len(starts) > maximum:
        starts = starts[np.linspace(0, len(starts) - 1, maximum, dtype=int)]
    return starts


def per_sample_errors(spec, predicted, target):
    e = spec.error_vector(predicted, target)
    physical = e * spec.err_scale
    return {
        "normalized_state_mse": (e**2).mean(-1),
        "root_position_squared_m": (physical[..., 0:3] ** 2).sum(-1),
        "root_orientation_squared_rad": (physical[..., 3:6] ** 2).sum(-1),
        "root_linear_velocity_squared_m_s": (physical[..., 6:9] ** 2).sum(-1),
        "root_angular_velocity_squared_rad_s": (physical[..., 9:12] ** 2).sum(-1),
        "joint_position_mse_rad": (physical[..., 12:41] ** 2).mean(-1),
        "joint_velocity_mse_rad_s": (physical[..., 41:70] ** 2).mean(-1),
    }


def aggregate(values, mask):
    out = {"samples": int(np.sum(mask))}
    for name, array in values.items():
        x = np.asarray(array)[mask]
        value = float(x.mean()) if len(x) else None
        if name != "normalized_state_mse":
            name = name.replace("_squared_", "_rmse_").replace("_mse_", "_rmse_")
            value = None if value is None else value**0.5
        out[name] = value
    return out


@torch.inference_mode()
def evaluate(wm, data, horizons, stride=8, maximum=64, device="cpu"):
    validate_trajectory(data)
    if not np.isclose(wm.spec.dt, float(data["dt"])):
        raise ValueError("World Model dt differs from recorded control dt")
    s = torch.as_tensor(data["states"], device=device)
    a = torch.as_tensor(data["actions"], device=device)
    xi = torch.as_tensor(data["xi"], device=device)
    # Teacher forcing resets to the real state at EVERY transition.
    teacher = []
    for start in range(0, len(a), 128):
        stop = min(start + 128, len(a))
        pred = wm.step(s[start:stop], a[start:stop], xi[start:stop])
        teacher.append(
            {
                k: v.cpu().numpy()
                for k, v in per_sample_errors(wm.spec, pred, s[start + 1 : stop + 1]).items()
            }
        )
    teacher = {k: np.concatenate([r[k] for r in teacher]) for k in teacher[0]}
    starts = window_starts(len(a), max(horizons), stride, maximum)
    # Same starts across all horizons; clean subset excludes any overflow in the full 32-step window.
    clean = np.array([not data["overflow"][i : i + max(horizons)].any() for i in starts])
    index = torch.as_tensor(starts, device=device)
    pred = s[index].clone()
    snapshots = {}
    for h in range(1, max(horizons) + 1):
        pred = wm.step(pred, a[index + h - 1], xi[index + h - 1])
        if h in horizons:
            snapshots[h] = {
                k: v.cpu().numpy()
                for k, v in per_sample_errors(wm.spec, pred, s[index + h]).items()
            }
    if not np.allclose(
        snapshots[1]["normalized_state_mse"],
        teacher["normalized_state_mse"][starts],
        rtol=1e-4,
        atol=1e-6,
    ):
        raise ValueError("Teacher-forced and autoregressive h=1 alignment failed")
    rows = []
    for subset, mask in [("all", np.ones(len(starts), dtype=bool)), ("no_overflow_32", clean)]:
        for h in horizons:
            rows.append(
                {
                    "protocol": "autoregressive",
                    "subset": subset,
                    "horizon": h,
                    "horizon_seconds": h * wm.spec.dt,
                    **aggregate(snapshots[h], mask),
                }
            )
    for subset, mask in [
        ("all", np.ones(len(a), dtype=bool)),
        ("no_overflow_transition", ~data["overflow"].astype(bool)),
    ]:
        rows.append(
            {
                "protocol": "teacher_forced",
                "subset": subset,
                "horizon": 1,
                "horizon_seconds": wm.spec.dt,
                **aggregate(teacher, mask),
            }
        )
    detail = {
        "window_starts": starts,
        "clean_windows": clean,
        **{"teacher_" + k: v for k, v in teacher.items()},
        **{f"ar_h{h}_" + k: v for h, metrics in snapshots.items() for k, v in metrics.items()},
    }
    return rows, detail
