"""CPU-only synthetic fixtures in temporary directories, never experimental results."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_logs
import monitor
import training
from torch.utils.tensorboard import SummaryWriter


class LoggingTests(unittest.TestCase):
    def test_no_training_does_not_create_directory_or_fake_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "full_seed0"
            result = analyze_logs.summarize(run)
            self.assertEqual(result["status"], "not_started")
            self.assertFalse(run.exists())
            self.assertEqual(result["figures"], [])

    def test_instrumented_cli_keeps_original_preset_and_flags(self):
        cfg = training.make_config("full")
        native = training.full_command(cfg)
        observed = training.monitored_command(cfg)
        self.assertEqual(observed[observed.index("--") + 1 :], native[1:])
        self.assertNotIn("--resume", observed)
        self.assertNotIn("--init-from", observed)

    def test_writer_flushes_without_duplicate_scalar_and_records_nonfinite(self):
        from torch.utils import tensorboard

        original = tensorboard.SummaryWriter
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            with (
                patch.dict("os.environ", {"HF_TOKEN": "test_secret_never_export"}),
                monitor.TrainingMonitor(run, sample_memory=False),
            ):
                writer = tensorboard.SummaryWriter(str(run / "tb"))
                self.assertEqual(writer.flush_secs, 15)
                writer.add_scalar("wm/loss", 0.5, 50)
                writer.add_scalar("wm/loss", float("nan"), 100)
                writer.add_scalar("time/env_steps", 290, 100)
                writer.add_scalar("time/policy_updates", 0, 100)
            rows = analyze_logs.scalar_rows(run)
            self.assertEqual(len([r for r in rows if r["tag"] == "wm/loss"]), 2)
            self.assertEqual(
                json.loads((run / "monitor_status.json").read_text())["nonfinite_scalar_count"], 1
            )
            self.assertNotIn("test_secret_never_export", (run / "monitor_status.json").read_text())
        self.assertIs(tensorboard.SummaryWriter, original)

    def test_interruption_flushes_and_marks_status(self):
        from torch.utils import tensorboard

        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            with (
                self.assertRaises(KeyboardInterrupt),
                monitor.TrainingMonitor(run, sample_memory=False),
            ):
                writer = tensorboard.SummaryWriter(str(run / "tb"))
                writer.add_scalar("wm/loss", 0.2, 50)
                raise KeyboardInterrupt
            self.assertEqual(
                json.loads((run / "monitor_status.json").read_text())["status"], "interrupted"
            )
            self.assertEqual(len(analyze_logs.scalar_rows(run)), 1)

    def test_no_events_means_no_curves_and_unknown_iterations(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "run_metadata.json").write_text(
                json.dumps({"status": "failed", "pretrained_checkpoint_used": False})
            )
            result = analyze_logs.summarize(run)
            self.assertEqual(result["figures"], [])
            self.assertIsNone(result["actual_iterations"])
            self.assertFalse((run / "figures").exists())
            self.assertTrue((run / "TRAINING_REPORT.md").is_file())

    def test_smoothing_does_not_bridge_nonfinite_values(self):
        values = analyze_logs.trailing_mean([1, 3, float("nan"), 9, 11], 2)
        self.assertEqual(values[:2], [1, 2])
        self.assertTrue(math.isnan(values[2]))
        self.assertEqual(values[3:], [9, 10])

    def test_throughput_uses_native_console_instead_of_duplicate_calculation(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "console.log").write_text("it 50 | 0.01 h | 2.00 it/s | env steps 165 | extra\n")
            rows = analyze_logs.throughput_rows(run, [])
            self.assertEqual(rows[0]["value"], 2)
            self.assertEqual(rows[0]["source"], "community console it/s")

    def test_real_event_format_produces_five_plot_pairs_and_preserves_raw_points(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "requested_config.json").write_text(
                json.dumps(
                    {
                        "seed": 0,
                        "resume": None,
                        "schedule": {"total_iterations": 86000, "wm_warmup_iterations": 500},
                    }
                )
            )
            (run / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "interrupted",
                        "pretrained_checkpoint_used": False,
                        "resume": None,
                        "init_from": None,
                    }
                )
            )
            with SummaryWriter(str(run / "tb")) as writer:
                for index, step in enumerate([500, 550, 600]):
                    for tag in [
                        "wm/loss",
                        "wm/online_nse_8step",
                        "policy/imag_objective",
                        "policy/critic_loss",
                    ]:
                        writer.add_scalar(tag, 1 / (index + 1), step)
                    writer.add_scalar("time/hours", (index + 1) / 3600, step)
                    writer.add_scalar("time/env_steps", step * 2.5 + 40, step)
            result = analyze_logs.summarize(run, inspect_checkpoint=False)
            self.assertEqual(len(result["figures"]), 10)
            self.assertEqual(result["world_model_only_boundary"], 500)
            self.assertTrue(result["interrupted"])
            self.assertFalse(result["training_completed_normally"])
            with (run / "metrics/scalars.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 18)
            self.assertTrue((run / "training_summary.json").exists())

    def test_final_checkpoint_verifies_completed_iterations_and_optimizer_counts(self):
        import torch

        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cfg = {
                "seed": 0,
                "resume": None,
                "init_from": None,
                "schedule": {"total_iterations": 12, "wm_warmup_iterations": 4},
            }
            (run / "config.json").write_text(json.dumps(cfg))
            (run / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "pretrained_checkpoint_used": False,
                        "resume": None,
                        "init_from": None,
                    }
                )
            )
            ck = {
                "iteration": 12,
                "env_steps": 70,
                "num_updates": 8,
                "train_seconds": 2.0,
                "config": cfg,
            }
            for name in ["wm", "policy", "critic"]:
                ck[name] = {"weight": torch.ones(1)}
                ck[name + "_opt"] = {
                    "state": {0: {"step": torch.tensor(12 if name == "wm" else 8)}}
                }
            torch.save(ck, run / "ckpt_latest.pt")
            result = analyze_logs.summarize(run)
            self.assertTrue(result["training_completed_normally"])
            self.assertEqual(result["actual_iterations"], 12)
            self.assertEqual(result["actual_environment_steps"], 70)
            self.assertEqual(result["checkpoint"]["optimizer_steps"]["wm_opt"], 12)
            self.assertEqual(result["figures"], [])

    def test_library_rebuild_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "prepared_data/bones_paper.pt"
            output.parent.mkdir()
            output.write_bytes(b"preserved")
            with (
                patch.object(training, "output_root", return_value=root),
                patch.object(training, "run_child") as launch,
            ):
                with self.assertRaises(FileExistsError):
                    training.rebuild_library()
                launch.assert_not_called()
            self.assertEqual(output.read_bytes(), b"preserved")

    def test_library_rebuild_refuses_original_path_via_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "original/bones_paper.pt"
            old.parent.mkdir()
            old.write_bytes(b"original unchanged")
            (root / "prepared_data").symlink_to(old.parent, target_is_directory=True)
            with (
                patch.object(training, "output_root", return_value=root),
                patch.object(training, "DEFAULT_LIBRARY", old),
                patch.object(training, "run_child") as launch,
            ):
                with self.assertRaises(ValueError):
                    training.rebuild_library()
                launch.assert_not_called()
            self.assertEqual(old.read_bytes(), b"original unchanged")

    def test_duplicate_restart_points_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            for suffix, when in [("_first", 1000), ("_resumed", 2000)]:
                with SummaryWriter(str(run / "tb"), filename_suffix=suffix) as writer:
                    writer.add_scalar("wm/loss", 1, 100, walltime=when)
            with patch.object(analyze_logs, "draw", return_value=[]):
                result = analyze_logs.summarize(run, inspect_checkpoint=False)
            self.assertTrue(result["step_restart_detected"])
            self.assertEqual(result["metric_summary"]["wm/loss"]["samples"], 2)


if __name__ == "__main__":
    unittest.main()
