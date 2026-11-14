"""CPU tests only: math, archived metadata and installed source interfaces."""

import ast
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from metrics import compare_outcome, control_timing, root_errors, video_window
from replay import ROOT, create_case_output, load_config, preflight, readable_scene_archive


class ReplayTests(unittest.TestCase):
    def test_reusing_viewer_id_preserves_legacy_and_previous_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {"output_dir": directory}
            legacy = Path(directory) / "viewer/demo/motion"
            legacy.mkdir(parents=True)
            report = legacy / "session.json"
            report.write_text("legacy")
            first = create_case_output(cfg, "viewer", "motion", "demo")
            (first / "session.json").write_text("first")
            second = create_case_output(cfg, "viewer", "motion", "demo")
            self.assertNotEqual(first, second)
            self.assertEqual(report.read_text(), "legacy")
            self.assertEqual((first / "session.json").read_text(), "first")
            for mode in ["recordings", "renders"]:
                create_case_output(cfg, mode, "motion", "demo")
                with self.assertRaisesRegex(FileExistsError, "new --run-id"):
                    create_case_output(cfg, mode, "motion", "demo")

    def test_scene_archive_repairs_only_empty_defaults_and_adds_missing_assets(self):
        xml = b'<mujoco><default><default/><default class="keep"><geom friction="1"/></default></default><asset><mesh name="m" file="mesh.STL"/></asset></mujoco>'
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            archive_path = folder / "scene.zip"
            (folder / "mesh.STL").write_bytes(b"original mesh bytes")
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("scene.xml", xml)
                archive.writestr("existing.bin", b"unchanged")
            before = archive_path.read_bytes()
            with (
                readable_scene_archive(archive_path, asset_dir=folder) as buffer,
                zipfile.ZipFile(buffer) as archive,
            ):
                tree = ET.fromstring(archive.read("scene.xml"))
                defaults = tree.find("default")
                self.assertEqual(len(defaults), 1)
                self.assertEqual(defaults[0].get("class"), "keep")
                self.assertEqual(defaults[0][0].get("friction"), "1")
                self.assertEqual(archive.read("mesh.STL"), b"original mesh bytes")
                self.assertEqual(archive.read("existing.bin"), b"unchanged")
            self.assertEqual(archive_path.read_bytes(), before)
            (folder / "mesh.STL").unlink()
            with self.assertRaises(FileNotFoundError):
                readable_scene_archive(archive_path, asset_dir=folder)

    def test_timing_uses_environment_config_not_exported_scene_defaults(self):
        # mjlab applies 5 ms to the compiled physics model; exported MjSpec retains 2 ms.
        timing = control_timing(0.005, 4)
        self.assertAlmostEqual(timing["control_dt_s"], 0.02)
        with self.assertRaises(ValueError):
            control_timing(0.002, 4)
        with self.assertRaises(ValueError):
            control_timing(float("nan"), 4)

    def test_authoritative_result_saved_before_scene_export(self):
        tree = ast.parse(
            (ROOT / "my_work/experiments/exp_004_motion_visualization/replay.py").read_text()
        )
        record = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "record_case"
        )
        persisted = next(
            n.lineno
            for n in ast.walk(record)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "write_json"
        )
        exported = next(
            n.lineno
            for n in ast.walk(record)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "to_zip"
        )
        self.assertLess(persisted, exported)

    def test_metadata_matches_archives_without_gpu_imports(self):
        cfg = load_config()
        status = preflight(cfg)
        self.assertFalse(status["simulation_started"])
        self.assertFalse(status["checkpoint_loaded"])
        self.assertEqual(
            [c["motion"] for c in cfg["cases"]],
            ["walk1_subject1", "dance2_subject4", "fallAndGetUp1_subject1"],
        )
        self.assertEqual(cfg["cases"][1]["historical_fail_time_s"], 60.92)
        self.assertEqual(cfg["cases"][2]["historical_fail_time_s"], 7.98)
        for package in ["torch", "mujoco", "warp", "mjlab"]:
            self.assertNotIn(package, sys.modules)

    def test_quaternion_double_cover_and_translation(self):
        ref = np.array([[0, 0, 1, 1, 0, 0, 0]], dtype=float)
        sim = np.array([[3, 4, 1, -1, 0, 0, 0]], dtype=float)
        result = root_errors(sim, ref)
        self.assertAlmostEqual(result["position_error_m"][0], 5)
        self.assertAlmostEqual(result["orientation_error_rad"][0], 0)
        self.assertAlmostEqual(result["benchmark_tilt_error"][0], 0)

    def test_yaw_error_is_not_benchmark_tilt_error(self):
        ref = np.array([[0, 0, 1, 1, 0, 0, 0]], dtype=float)
        sim = np.array([[0, 0, 1, np.sqrt(0.5), 0, 0, np.sqrt(0.5)]])
        result = root_errors(sim, ref)
        self.assertAlmostEqual(result["orientation_error_rad"][0], np.pi / 2)
        self.assertAlmostEqual(result["benchmark_tilt_error"][0], 0)

    def test_matching_ground_pose_does_not_fail(self):
        pose = np.array([[0, 0, 0.1, 0, 1, 0, 0]], dtype=float)
        result = root_errors(pose, pose)
        self.assertEqual(result["height_error_m"][0], 0)
        self.assertEqual(result["benchmark_tilt_error"][0], 0)

    def test_bad_pose_rejected(self):
        with self.assertRaises(ValueError):
            root_errors(np.zeros((2, 7)), np.zeros((2, 7)))
        with self.assertRaises(ValueError):
            root_errors(np.zeros((2, 6)), np.zeros((2, 6)))

    def test_pending_and_mismatched_outcomes(self):
        old = {"failed": True, "fail_step": 399}
        self.assertEqual(compare_outcome(old, None, -1)["status"], "not_measured")
        self.assertEqual(compare_outcome(old, True, 399)["status"], "exact_outcome_match")
        new = compare_outcome(old, True, 401)
        self.assertEqual(new["status"], "mismatch")
        self.assertAlmostEqual(new["fail_time_difference_s"], 0.04)
        self.assertEqual(compare_outcome(old, False, -1)["status"], "mismatch")

    def test_short_video_ends_before_terminal_state(self):
        self.assertEqual(video_window(3046, 3046, seconds=12), (2446, 3046))
        self.assertEqual(video_window(399, 399, seconds=12), (0, 399))
        self.assertEqual(video_window(13066, -1, seconds=12), (0, 600))
        with self.assertRaises(ValueError):
            video_window(399, 399, start=8)
        with self.assertRaises(ValueError):
            video_window(0, -1)

    def test_existing_community_interfaces_without_importing_them(self):
        tree = ast.parse((ROOT / "tools/render_rollouts.py").read_text())
        pane = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Pane")
        methods = {n.name for n in pane.body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"__init__", "render", "close"} <= methods)
        bench = ast.parse((ROOT / "src/supermimic/eval/benchmark.py").read_text())
        run = next(
            n for n in bench.body if isinstance(n, ast.FunctionDef) and n.name == "run_benchmark"
        )
        self.assertTrue({"recording", "terminate", "num_envs"} <= {n.arg for n in run.args.args})
        site = ROOT / ".venv/lib/python3.12/site-packages"
        viewer = ast.parse((site / "mjlab/viewer/viser/viewer.py").read_text())
        cls = next(
            n for n in viewer.body if isinstance(n, ast.ClassDef) and n.name == "ViserPlayViewer"
        )
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        self.assertIn("viser_server", {n.arg for n in init.args.args})
        mu = ast.parse((site / "mujoco/__init__.py").read_text())
        self.assertTrue(
            {"to_zip", "from_zip"} <= {n.name for n in mu.body if isinstance(n, ast.FunctionDef)}
        )


if __name__ == "__main__":
    unittest.main()
