"""CPU/static checks only. Never construct Trainer or launch a training command."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
import training

from supermimic.presets import release


class TrainingPreparationTests(unittest.TestCase):
    def test_full_preserves_release_recipe(self):
        expected = asdict(release())
        actual = asdict(training.make_config("full"))
        for key in ["run_name", "log_root"]:
            expected.pop(key)
            actual.pop(key)
        self.assertEqual(expected, actual)
        self.assertEqual(actual["schedule"]["total_iterations"], 86000)
        self.assertEqual(actual["schedule"]["wm_warmup_iterations"], 500)

    def test_no_pretrained_initialization_or_resume_flags(self):
        for mode in ["full", "smoke"]:
            cfg = training.make_config(mode)
            self.assertIsNone(cfg.resume)
            self.assertIsNone(cfg.init_from)
        command = training.full_command(training.make_config("full"))
        self.assertNotIn("--resume", command)
        self.assertNotIn("--init-from", command)
        self.assertNotIn("realmimic_g1_86k.pt", " ".join(command))
        cfg = training.make_config("full")
        cfg.init_from = Path("weights/model.pt")
        with self.assertRaises(ValueError):
            training.assert_fresh(cfg)

    def test_smoke_covers_both_updates_without_changing_architecture(self):
        cfg = training.make_config("smoke")
        full = release()
        self.assertEqual(cfg.world_model.hidden, full.world_model.hidden)
        self.assertEqual(cfg.policy.encoder_hidden, full.policy.encoder_hidden)
        self.assertEqual(cfg.policy.decoder_hidden, full.policy.decoder_hidden)
        self.assertGreater(cfg.schedule.total_iterations, cfg.schedule.wm_warmup_iterations)
        self.assertEqual(cfg.schedule.total_iterations - cfg.schedule.wm_warmup_iterations, 8)
        self.assertIsNone(cfg.data.library)
        self.assertEqual(cfg.data.npz_dir, Path("data/npz/bones_seed"))
        self.assertIsNone(cfg.wm_trainer.compile)
        self.assertIsNone(cfg.policy_trainer.compile)

    def test_corrupted_library_is_rejected_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "broken.pt"
            p.write_bytes(b"PK\x03\x04truncated")
            before = p.read_bytes()
            report = training.inspect_library(p, ["a"], {"a": 10})
            self.assertFalse(report["ready"])
            self.assertEqual(before, p.read_bytes())

    def test_library_names_ranges_and_shape_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "small.pt"
            ref = torch.zeros(20, 71, dtype=torch.float16)
            ref[:, 3] = 1
            data = {"names": ["a", "b"], "starts": [0, 10], "lengths": [10, 10], "ref": ref}
            torch.save(data, p)
            self.assertTrue(training.inspect_library(p, ["a", "b"], {"a": 10, "b": 10})["ready"])
            data["starts"][1] = 11
            torch.save(data, p)
            self.assertFalse(training.inspect_library(p, ["a", "b"], {"a": 10, "b": 10})["ready"])

    def test_full_does_not_launch_when_library_is_bad(self):
        report = {
            "smoke_data_ready": True,
            "full_data_ready": False,
            "library": {"error": "corrupt library"},
        }
        with (
            patch.object(training, "preflight", return_value=report),
            patch.object(training.subprocess, "run") as launch,
        ):
            with self.assertRaisesRegex(RuntimeError, "corrupt library"):
                training.run_training("full", 0)
            launch.assert_not_called()

    def test_full_does_not_launch_without_actual_smoke_success(self):
        report = {"smoke_data_ready": True, "full_data_ready": True}
        with (
            patch.object(training, "preflight", return_value=report),
            patch.object(training, "smoke_passed", return_value=False),
            patch.object(training.subprocess, "run") as launch,
        ):
            with self.assertRaisesRegex(RuntimeError, "Smoke Test"):
                training.run_training("full", 0)
            launch.assert_not_called()

    def test_non_default_library_only_changes_data_source(self):
        cfg = training.make_config(
            "full",
            seed=5,
            library=Path("my_work/outputs/exp_007_full_training/prepared_data/bones_paper.pt"),
        )
        self.assertEqual(cfg.seed, 5)
        self.assertEqual(cfg.run_name, "full_seed5")
        self.assertIn(str(cfg.data.library), training.full_command(cfg))
        self.assertEqual(cfg.env.num_envs, 4096)


if __name__ == "__main__":
    unittest.main()
