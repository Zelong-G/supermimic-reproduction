"""Small CPU-only tests: state/action alignment and rollout protocol separation."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from analysis import evaluate, validate_trajectory, window_starts

from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


class ExactIntegrator:
    def __init__(self, gain=1):
        self.spec = StateSpec(StateSpecCfg(), "cpu")
        self.gain = gain

    def step(self, s, a, xi):
        out = s.clone()
        out[..., 0] += self.gain * a[..., 0]
        return out


def data(length=145):
    s = np.zeros((length + 1, 71), dtype=np.float32)
    s[:, 3] = 1
    a = np.zeros((length, 29), dtype=np.float32)
    a[:, 0] = 0.01
    s[:, 0] = np.arange(length + 1) * 0.01
    return {
        "states": s,
        "actions": a,
        "xi": np.zeros((length, 33), dtype=np.float32),
        "terminated": np.zeros(length, dtype=bool),
        "truncated": np.zeros(length, dtype=bool),
        "overflow": np.zeros(length, dtype=bool),
        "dt": np.array(0.02),
    }


class AnalysisTests(unittest.TestCase):
    def test_exact_model_and_final_teacher_batch_alignment(self):
        rows, _ = evaluate(ExactIntegrator(), data(), [1, 4, 8, 16, 32])
        for row in rows:
            self.assertLess(row["normalized_state_mse"], 1e-9)
        tf = next(r for r in rows if r["protocol"] == "teacher_forced" and r["subset"] == "all")
        self.assertEqual(tf["samples"], 145)

    def test_autoregression_accumulates_error_teacher_forcing_does_not(self):
        rows, _ = evaluate(ExactIntegrator(2), data(), [1, 4, 8, 16, 32])
        ar = {
            r["horizon"]: r
            for r in rows
            if r["protocol"] == "autoregressive" and r["subset"] == "all"
        }
        self.assertAlmostEqual(
            ar[32]["normalized_state_mse"] / ar[1]["normalized_state_mse"], 1024, delta=0.1
        )

    def test_exclude_overflow_from_whole_window(self):
        d = data()
        d["overflow"][20] = True
        rows, detail = evaluate(ExactIntegrator(), d, [1, 4, 8, 16, 32])
        clean = [r for r in rows if r["subset"] == "no_overflow_32"]
        self.assertEqual(len({r["samples"] for r in clean}), 1)
        self.assertLess(clean[0]["samples"], len(detail["window_starts"]))

    def test_no_reset_crossing_and_timestep_mismatch(self):
        d = data()
        d["terminated"][20] = True
        with self.assertRaises(ValueError):
            validate_trajectory(d)
        d = data()
        d["dt"] = np.array(0.01)
        with self.assertRaises(ValueError):
            validate_trajectory(d)

    def test_true_terminal_endpoint_is_valid(self):
        d = data()
        d["terminated"][-1] = True
        validate_trajectory(d)

    def test_no_valid_windows(self):
        with self.assertRaises(ValueError):
            window_starts(31)

    def test_state_quaternion_and_missing_velocity_rejected(self):
        d = data()
        d["states"][:, 3] = 0
        with self.assertRaises(ValueError):
            validate_trajectory(d)
        d = data()
        d["states"] = d["states"][:, :36]
        with self.assertRaises(ValueError):
            validate_trajectory(d)


if __name__ == "__main__":
    unittest.main()
