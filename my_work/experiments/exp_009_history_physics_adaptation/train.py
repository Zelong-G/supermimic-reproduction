"""Manual fine-tuning entry point. Default mode is inspect; validate is one scratch update only."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import random
import signal
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from common import (
    OUT,
    ROOT,
    build_config,
    file_hash,
    output_path,
    protected_sources,
    save_json,
    settings,
    state_hash,
)
from joint import JointTrainer
from models import AdaptivePolicy

from supermimic.training.loop import Trainer
from supermimic.training.schedule import scheduled_lr


class Experiment(Trainer):
    def __init__(self, run_name, lambda_mu=1.0, lambda_action=1.0):
        target = output_path(OUT / run_name)
        if target.exists():
            raise FileExistsError(f"{target} exists; use a new run name or --resume")
        ck = torch.load(ROOT / settings()["pretrained"], map_location="cpu", weights_only=False)
        cfg = build_config(ck, run_name)
        del ck
        super().__init__(cfg)
        if self.world != 1:
            raise ValueError("Experiment009 currently supports one GPU only")
        base = self.policy.requires_grad_(False).eval()
        self.wm.requires_grad_(False).eval()
        self.frozen_hashes = {"policy": state_hash(base), "world_model": state_hash(self.wm)}
        self.policy = AdaptivePolicy(base, settings()["alpha"]).to(self.device)
        # No World Model warm-up, normalizer updates or optimizer is retained.
        del self.wm_trainer, self.policy_trainer
        self.policy_trainer = JointTrainer(
            self.wm,
            self.policy,
            self.critic,
            self.lib,
            cfg.policy_trainer,
            lambda_mu,
            lambda_action,
        )
        self._obs, _ = self.env.reset()
        self._finished_returns = []
        self._finished_lens = []
        self.history = []
        self.stop_requested = False
        assert self.env.step_dt == 0.02
        save_json(
            self.log_dir / "experiment.json",
            {
                "settings": settings(),
                "frozen_hashes": self.frozen_hashes,
                "resolved_config": dataclasses.asdict(cfg),
                "critic_updated": True,
                "wm_optimizer": False,
                "normalization": "Frozen base observation normalization is reused by new encoder/adapter; original policy normalization never updated.",
            },
        )

    @torch.no_grad()
    def act(self, obs):
        return self.policy(obs["policy"], obs["goal"])

    def assert_frozen(self):
        got = {"policy": state_hash(self.policy.base), "world_model": state_hash(self.wm)}
        if got != self.frozen_hashes:
            raise AssertionError(f"Frozen state changed: {got}")
        assert all(not p.requires_grad and p.grad is None for p in self.policy.base.parameters())
        assert all(not p.requires_grad and p.grad is None for p in self.wm.parameters())
        return got

    def fill_replay(self):
        for _ in range(self.cfg.schedule.warmup_env_steps):
            self._env_step()

    def save_experiment(self, tag=None):
        self.assert_frozen()
        name = tag or f"ckpt_{self.iteration:07d}.pt"
        path = output_path(self.log_dir / name)
        if path.exists():
            raise FileExistsError(path)
        payload = {
            "format": "exp009/v1",
            "iteration": self.iteration,
            "iteration_complete": getattr(self, "_iteration_complete", True),
            "env_steps": self.env_steps,
            "step_acc": self._step_acc,
            "learned": self.policy.learned_state(),
            "critic": self.critic.state_dict(),
            "joint_opt": self.policy_trainer.opt.state_dict(),
            "critic_opt": self.policy_trainer.critic_opt.state_dict(),
            "frozen_hashes": self.frozen_hashes,
            "pretrained": settings()["pretrained"],
            "pretrained_sha256": file_hash(ROOT / settings()["pretrained"]),
            "settings": settings(),
            "lambda_mu": self.policy_trainer.lambda_mu,
            "lambda_action": self.policy_trainer.lambda_action,
            "config": dataclasses.asdict(self.cfg),
            "sampler": {
                k: getattr(self.command, k).clone()
                for k in ["bin_return", "bin_count", "bin_memory"]
            },
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
            "metrics": self.history,
        }
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        save_json(
            self.log_dir / "latest.json",
            {"checkpoint": str(path.relative_to(ROOT)), "iteration": self.iteration},
        )
        return path

    def resume_experiment(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if not ck.get("iteration_complete", True):
            raise ValueError(
                "Partial-iteration diagnostic snapshot: resume the last completed numbered checkpoint"
            )
        if ck["frozen_hashes"] != self.frozen_hashes:
            raise ValueError("Resume pretrained state differs")
        if ck["settings"] != settings():
            raise ValueError("Resume settings changed; preserve the original joint experiment")
        self.policy.load_learned(ck["learned"])
        self.critic.load_state_dict(ck["critic"])
        t = self.policy_trainer
        t.opt.load_state_dict(ck["joint_opt"])
        t.critic_opt.load_state_dict(ck["critic_opt"])
        t.lambda_mu = ck["lambda_mu"]
        t.lambda_action = ck["lambda_action"]
        self.iteration = ck["iteration"]
        self.env_steps = ck["env_steps"]
        self._step_acc = ck["step_acc"]
        self.history = ck["metrics"]
        for k, v in ck["sampler"].items():
            getattr(self.command, k).copy_(v)
        self.command._bin_count_synced.copy_(self.command.bin_count)
        torch.set_rng_state(ck["rng_torch"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in ck["rng_cuda"]])
        np.random.set_state(ck["rng_numpy"])
        random.setstate(ck["rng_python"])
        # As upstream, the environment and replay are refilled, not bitwise trajectory continuation.
        self.assert_frozen()

    def run_joint(self, max_hours=None, *, deadline=None):
        self.fill_replay()
        if self.iteration == 0 and not (self.log_dir / "ckpt_0000000.pt").exists():
            self.save_experiment()
        start = time.perf_counter()
        c = self.cfg
        sch = c.schedule
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop_requested", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop_requested", True))
        best = -float("inf")
        stale = 0
        # Reconstruct the training-only early-stop state on resume.
        stopping = settings()
        for previous in self.history:
            if previous["imag_objective"] > best + stopping["early_stopping_min_delta"]:
                best = previous["imag_objective"]
                stale = 0
            else:
                stale += 1
        if max_hours is None:
            max_hours = sch.max_hours
        try:
            while self.iteration < sch.total_iterations:
                self._iteration_complete = False
                tick = time.perf_counter()
                self._step_acc += 1 / sch.updates_per_env_step
                steps = int(self._step_acc)
                self._step_acc -= steps
                for _ in range(steps):
                    self._env_step()
                lr = scheduled_lr(
                    self.iteration,
                    c.policy_trainer.lr,
                    sch.lr_final_ratio,
                    sch.lr_decay_iterations,
                    sch.lr_restart_iteration,
                    sch.lr_restart_ratio,
                    sch.lr_restart_decay_iterations,
                )
                self.policy_trainer.set_lr(lr)
                self.policy_trainer.capture_predictions = self.iteration % sch.log_interval == 0
                torch.cuda.reset_peak_memory_stats()
                m = self.policy_trainer.update(self.buffer)
                torch.cuda.synchronize()
                self.iteration += 1
                self._iteration_complete = True
                m.update(
                    iteration=self.iteration,
                    lr=lr,
                    seconds_per_iteration=time.perf_counter() - tick,
                    elapsed_seconds=time.perf_counter() - start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    env_steps=self.env_steps,
                )
                m["imagined_steps_per_second"] = (
                    c.policy_trainer.batch_size
                    * c.policy_trainer.horizon
                    * c.policy_trainer.grad_accumulation
                    / m["seconds_per_iteration"]
                )
                m.update({f"overflow_{k}": v for k, v in self._overflow_fractions().items()})
                self.history.append(m)
                if self.policy_trainer.capture_predictions:
                    with (self.log_dir / "friction_predictions.jsonl").open("a") as f:
                        for pred in self.policy_trainer.last_predictions:
                            f.write(json.dumps({"iteration": self.iteration, **pred}) + "\n")
                csv_path = self.log_dir / "metrics.csv"
                with csv_path.open("a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(m))
                    if f.tell() == 0:
                        w.writeheader()
                    w.writerow(m)
                if self.writer:
                    for k, v in m.items():
                        self.writer.add_scalar(k, v, self.iteration)
                if self.iteration % sch.log_interval == 0:
                    print(json.dumps(m), flush=True)
                if self.iteration % sch.checkpoint_interval == 0:
                    self.save_experiment()
                cfg = settings()
                patience = cfg["early_stopping_patience"]
                if m["imag_objective"] > best + cfg["early_stopping_min_delta"]:
                    best = m["imag_objective"]
                    stale = 0
                else:
                    stale += 1
                if (
                    patience is not None
                    and self.iteration >= cfg["early_stopping_min_iterations"]
                    and stale >= patience
                ):
                    break
                if (
                    self.stop_requested
                    or (max_hours is not None and (time.perf_counter() - start) / 3600 >= max_hours)
                    or (deadline is not None and time.perf_counter() >= deadline)
                ):
                    break
        except (KeyboardInterrupt, torch.cuda.OutOfMemoryError, FloatingPointError) as exc:
            save_json(
                self.log_dir / "interruption.json",
                {
                    "error": str(exc),
                    "iteration": self.iteration,
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "peak_allocated": torch.cuda.max_memory_allocated(),
                    "batch": 2048,
                    "horizon": 32,
                    "envs": 4096,
                },
            )
            self.save_experiment(f"interrupted_{self.iteration:07d}_{time.time_ns()}.pt")
            raise
        finally:
            if self.writer:
                self.writer.flush()
        if not (self.log_dir / f"ckpt_{self.iteration:07d}.pt").exists():
            self.save_experiment()
        return self.history


def inspect():
    import importlib.metadata

    ck = torch.load(ROOT / settings()["pretrained"], map_location="cpu", weights_only=False)
    c = build_config(ck, "full_seed0")
    path = Path(c.data.library)
    if not zipfile.is_zipfile(path):
        raise ValueError("Configured BONES-SEED packed library is unreadable")
    pack = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert len(pack["names"]) == 129491 and pack["ref"].shape == (47345789, 71)
    report = {
        "status": "inspected_not_fine_tuned",
        "checkpoint": settings()["pretrained"],
        "checkpoint_sha256": file_hash(ROOT / settings()["pretrained"]),
        "checkpoint_iteration": ck["iteration"],
        "config": dataclasses.asdict(c),
        "source_hashes": protected_sources(),
        "library": {
            "path": str(path.relative_to(ROOT)),
            "clips": len(pack["names"]),
            "frames": len(pack["ref"]),
            "bytes": path.stat().st_size,
            "original_library_readable": zipfile.is_zipfile(ROOT / "data/library/bones_paper.pt"),
            "reason": "Reuse the already validated Exp007 rebuild of the original filtered BONES-SEED library; original truncated archive is preserved.",
        },
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["torch", "mujoco", "mujoco-warp", "warp-lang"]
        },
        "full_training_started": False,
    }
    save_json(OUT / "inspection.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["source_hashes", "config"]}, indent=2
        )
    )


def configure_continuation(
    exp, checkpoint, *, total_iterations=None, max_hours=None, extend_hours=None
):
    """Run-local limits; the shared configuration and parent output stay intact."""
    saved = checkpoint["config"]["schedule"]
    exp.cfg.schedule.total_iterations = int(saved["total_iterations"])
    exp.cfg.schedule.max_hours = saved.get("max_hours")
    if extend_hours is not None:
        if extend_hours <= 0 or total_iterations is not None or max_hours is not None:
            raise ValueError("Use positive extend_hours without other limit overrides")
        exp.cfg.schedule.total_iterations = 1_000_000_000
        exp.cfg.schedule.max_hours = float(extend_hours)
    else:
        if total_iterations is not None:
            exp.cfg.schedule.total_iterations = int(total_iterations)
        if max_hours is not None:
            if max_hours <= 0:
                raise ValueError("max_hours must be positive")
            exp.cfg.schedule.max_hours = float(max_hours)
    if exp.cfg.schedule.total_iterations <= exp.iteration:
        raise ValueError("Total iterations must exceed the restored checkpoint iteration")
    return {
        "parent_iteration": exp.iteration,
        "iteration_ceiling": exp.cfg.schedule.total_iterations,
        "max_hours_per_invocation": exp.cfg.schedule.max_hours,
        "time_limited_extension": extend_hours is not None,
        "training_sizes": {
            "horizon": exp.cfg.policy_trainer.horizon,
            "batch": exp.cfg.policy_trainer.batch_size,
            "envs": exp.cfg.env.num_envs,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["inspect", "validate", "train"], nargs="?", default="inspect")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", type=Path)
    p.add_argument("--max-hours", type=float)
    p.add_argument("--total-iterations", type=int)
    p.add_argument(
        "--extend-hours",
        type=float,
        help="Continue in a fresh --run-name directory for this many training-loop hours; bypass the old iteration ceiling",
    )
    args = p.parse_args()
    deadline = (
        time.perf_counter() + args.extend_hours * 3600 if args.extend_hours is not None else None
    )
    if args.extend_hours is not None and (args.resume is None or args.run_name is None):
        p.error("--extend-hours requires --resume and a fresh --run-name")
    if args.extend_hours is not None and (
        args.extend_hours <= 0 or args.total_iterations is not None or args.max_hours is not None
    ):
        p.error("Use a positive --extend-hours alone, without --max-hours/--total-iterations")
    torch.set_num_threads(2)
    if args.mode == "inspect":
        inspect()
        return
    if args.mode == "validate":
        from validate import validate

        validate(args.run_name or "validation")
        return
    calibration = json.loads((OUT / "validation/calibration.json").read_text())
    for key in ["lambda_mu", "lambda_action"]:
        if settings()[key] is not None:
            calibration[key] = float(settings()[key])
    check = json.loads((OUT / "validation/full_check.json").read_text())
    if check["status"] != "passed":
        raise RuntimeError("Full-size validation has not passed")
    if args.resume:
        # Build a fresh staging directory, then attach to the explicitly resumed run.
        resume_path = output_path(args.resume)
        name = args.run_name or f".resume_{time.time_ns()}"
        exp = Experiment(name, calibration["lambda_mu"], calibration["lambda_action"])
        exp.resume_experiment(resume_path)
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        control = configure_continuation(
            exp,
            saved,
            total_iterations=args.total_iterations,
            max_hours=args.max_hours,
            extend_hours=args.extend_hours,
        )
        del saved
        if args.run_name is None:
            if exp.writer:
                exp.writer.close()
            import shutil

            shutil.rmtree(exp.log_dir)
            exp.log_dir = resume_path.parent
            from torch.utils.tensorboard import SummaryWriter

            exp.writer = SummaryWriter(str(exp.log_dir / "tb"))
        else:
            control["parent_checkpoint"] = str(resume_path.relative_to(ROOT))
            save_json(exp.log_dir / "continuation.json", control)
            save_json(exp.log_dir / "config.json", dataclasses.asdict(exp.cfg))
            metadata = json.loads((exp.log_dir / "experiment.json").read_text())
            metadata.update(resolved_config=dataclasses.asdict(exp.cfg), continuation=control)
            save_json(exp.log_dir / "experiment.json", metadata)
            exp.save_experiment()
        # Preserve logs after the restored checkpoint as an archive; avoid duplicate
        # iteration rows when continuing from an older periodic snapshot.
        for filename in ["metrics.csv", "friction_predictions.jsonl"]:
            if args.run_name is not None:
                break
            path = exp.log_dir / filename
            if not path.exists():
                continue
            archive = path.with_name(path.name + f".before_resume_{time.time_ns()}")
            path.rename(archive)
            if filename.endswith(".csv"):
                with archive.open() as f:
                    rows = list(csv.DictReader(f))
                rows = [r for r in rows if int(r["iteration"]) <= exp.iteration]
                if rows:
                    with path.open("w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0]))
                        w.writeheader()
                        w.writerows(rows)
            else:
                rows = [
                    json.loads(line) for line in archive.read_text().splitlines() if line.strip()
                ]
                with path.open("w") as f:
                    for row in rows:
                        if row["iteration"] <= exp.iteration:
                            f.write(json.dumps(row) + "\n")
    else:
        exp = Experiment(
            args.run_name or "joint_seed0", calibration["lambda_mu"], calibration["lambda_action"]
        )
        if args.total_iterations is not None:
            exp.cfg.schedule.total_iterations = args.total_iterations
    try:
        exp.run_joint(args.max_hours, deadline=deadline)
    finally:
        exp.env.close()


if __name__ == "__main__":
    main()
