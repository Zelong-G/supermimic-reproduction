"""No simulation: validate list integrity, result coverage and early-stop safety."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze


def fixture():
    return {
        "summary": {
            "clips": 2,
            "failed": 1,
            "mean_norm_return": 0.65,
            "step_weighted_norm_return": 0.65,
            "failed_at_end": 0,
        },
        "clips": [
            {
                "name": "a",
                "frames": 100,
                "steps": 100,
                "failed": False,
                "fail_step": -1,
                "norm_return": 0.9,
                "overflow_steps": 0,
            },
            {
                "name": "b",
                "frames": 100,
                "steps": 50,
                "failed": True,
                "fail_step": 50,
                "norm_return": 0.4,
                "overflow_steps": 1,
            },
        ],
    }


class BenchmarkTests(unittest.TestCase):
    def test_summary_and_actual_failure_rate(self):
        result = analyze.summarize_result(fixture(), ["a", "b"])
        self.assertEqual(result["failure_rate"], 0.5)
        self.assertEqual(result["overflow_clips"], 1)

    def test_missing_or_duplicate_motion_rejected(self):
        d = fixture()
        with self.assertRaises(ValueError):
            analyze.summarize_result(d, ["a", "b", "c"])
        d["clips"][1]["name"] = "a"
        with self.assertRaises(ValueError):
            analyze.summarize_result(d, ["a", "b"])

    def test_fabricated_summary_or_failure_timing_rejected(self):
        d = fixture()
        d["summary"]["failed"] = 0
        with self.assertRaises(ValueError):
            analyze.summarize_result(d, ["a", "b"])
        d = fixture()
        d["clips"][1]["fail_step"] = 51
        with self.assertRaises(ValueError):
            analyze.summarize_result(d, ["a", "b"])

    def test_list_duplicate_blank_or_path_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "list.txt"
            for content in ["a\na\n", "a\n\nb\n", "../a\n", " a\n"]:
                p.write_text(content)
                with self.assertRaises(ValueError):
                    analyze.read_list(p)
            p.write_text("a\nb\n")
            self.assertEqual(analyze.read_list(p), ["a", "b"])

    def test_incomplete_preprocessing_never_launches_evaluator(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(analyze, "OUT", Path(directory)),
            patch.object(
                analyze, "preflight", return_value={"ready": False, "errors": ["Missing lists"]}
            ),
            patch.object(analyze.subprocess, "run") as launch,
        ):
            with self.assertRaisesRegex(RuntimeError, "Missing lists"):
                analyze.run_stage("rand2000", None)
            launch.assert_not_called()

    def test_sample_npz_header_and_fps_validation(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "sample.npz"
            arrays = {
                "fps": np.array([50]),
                "joint_pos": np.zeros((3, 29)),
                "joint_vel": np.zeros((3, 29)),
                "body_pos_w": np.zeros((3, 30, 3)),
                "body_quat_w": np.zeros((3, 30, 4)),
                "body_lin_vel_w": np.zeros((3, 30, 3)),
                "body_ang_vel_w": np.zeros((3, 30, 3)),
                "body_names": np.array(["pelvis"]),
                "joint_names": np.array(["joint"]),
            }
            np.savez(p, **arrays)
            analyze.validate_sample_npz(p)
            arrays["fps"] = np.array([30])
            np.savez(p, **arrays)
            with self.assertRaises(ValueError):
                analyze.validate_sample_npz(p)

    def test_full_refused_before_prior_stages_exist(self):
        from types import SimpleNamespace

        args = SimpleNamespace(device="cuda:0")
        status = {"ready": True, "gpu": [{"index": 0, "free_mib": 24000}]}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(analyze, "OUT", Path(directory)),
            patch.object(analyze, "preflight", return_value=status),
            patch.object(analyze.subprocess, "run") as launch,
        ):
            with self.assertRaisesRegex(RuntimeError, "rand2000/hard"):
                analyze.run_stage("full", args)
            launch.assert_not_called()

    def test_existing_result_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "rand2000.json"
            p.write_text("preserved")
            with patch.object(analyze, "OUT", root), patch.object(analyze, "preflight") as check:
                with self.assertRaises(FileExistsError):
                    analyze.run_stage("rand2000", None)
                check.assert_not_called()
            self.assertEqual(p.read_text(), "preserved")


if __name__ == "__main__":
    unittest.main()
