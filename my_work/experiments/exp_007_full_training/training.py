"""Prepare release training, guarded fresh runs, smoke verification and TensorBoard export."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROFILE = HERE / "profiles.json"
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_LIBRARY = Path("data/library/bones_paper.pt")
EXPECTED_MANIFEST_MD5 = "5bc2aae2efc6f986ab4cf7adbb4d8ffc"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    from monitor import atomic_json

    atomic_json(path, value)


def digest(path, kind="sha256"):
    h = hashlib.new(kind)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def output_root():
    return ROOT / json.loads(PROFILE.read_text())["output_root"]


def make_config(mode, seed=0, library=None):
    from supermimic.presets import release

    profiles = json.loads(PROFILE.read_text())
    cfg = release()
    for dotted, value in profiles[mode].items():
        obj = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        key = parts[-1]
        if not hasattr(obj, key):
            raise ValueError(f"Unknown configuration field {dotted}")
        if dotted in ["data.library", "data.npz_dir", "data.manifest"] and value is not None:
            value = Path(value)
        setattr(obj, key, value)
    cfg.log_root = Path(profiles["output_root"])
    cfg.run_name = f"{mode}_seed{seed}"
    cfg.seed = seed
    if mode == "full" and library is not None:
        cfg.data.library = Path(library)
    assert_fresh(cfg)
    return cfg


def assert_fresh(cfg):
    if cfg.resume is not None or cfg.init_from is not None:
        raise ValueError("Fresh training requires resume=None and init_from=None")
    if cfg.seed < 0:
        raise ValueError("Seed must be nonnegative")
    if Path(cfg.log_root).is_absolute():
        raise ValueError("Expected workspace-relative output root")
    target = (ROOT / cfg.log_root / cfg.run_name).resolve()
    if not target.is_relative_to(output_root().resolve()):
        raise ValueError("Output escapes Experiment 007")


def full_command(cfg):
    assert_fresh(cfg)
    return [
        str(ROOT / ".venv/bin/sm-train"),
        "release",
        "--run-name",
        cfg.run_name,
        "--log-root",
        str(cfg.log_root),
        "--seed",
        str(cfg.seed),
        "--device",
        cfg.device,
        "--data.library",
        str(cfg.data.library),
    ]


def inspect_library(path, expected_names, expected_lengths):
    import torch

    report = {
        "path": str(path),
        "exists": path.is_file(),
        "ready": False,
        "scope": "ZIP readability, all names/lengths/contiguous ranges, fp16 71-D tensor shape, 64 sampled boundary rows; not a full finite/CRC scan.",
    }
    if not path.is_file():
        report["error"] = "Motion Library file is missing"
        return report
    stat = path.stat()
    report.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if not zipfile.is_zipfile(path):
        report["error"] = (
            "Packed library has no readable ZIP central directory; preserve it and rebuild to a new output path."
        )
        return report
    try:
        pack = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        names = pack["names"]
        starts = pack["starts"]
        lengths = pack["lengths"]
        ref = pack["ref"]
        if list(names) != expected_names or len(set(names)) != len(expected_names):
            raise ValueError("Library names differ from the full filtered set/order")
        if len(starts) != len(names) or len(lengths) != len(names):
            raise ValueError("Incomplete clip ranges")
        cursor = 0
        for name, start, length in zip(names, starts, lengths):
            if int(start) != cursor or int(length) != expected_lengths[name]:
                raise ValueError("Noncontiguous or incorrect frame ranges")
            cursor += int(length)
        if ref.shape != (cursor, 71) or ref.dtype != torch.float16:
            raise ValueError("Unexpected reference shape or dtype")
        indices = sorted({int(starts[i * (len(names) - 1) // 63]) for i in range(64)})
        if not torch.isfinite(ref[indices]).all():
            raise ValueError("Nonfinite sampled reference states")
        if not torch.allclose(
            ref[indices, 3:7].float().norm(dim=-1), torch.ones(len(indices)), atol=0.005
        ):
            raise ValueError("Invalid sampled root quaternions")
        report.update(
            ready=True,
            clips=len(names),
            frames=cursor,
            reference_shape=list(ref.shape),
            dtype=str(ref.dtype),
            sampled_rows=len(indices),
        )
    except (RuntimeError, ValueError, KeyError, OSError, EOFError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def runtime_info():
    import torch

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    gpus = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            index, name, total, used, free, driver = [p.strip() for p in line.split(",")]
            gpus.append(
                {
                    "index": int(index),
                    "name": name,
                    "total_mib": int(total),
                    "used_mib": int(used),
                    "free_mib": int(free),
                    "driver": driver,
                }
            )
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpus": gpus,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["mujoco", "mujoco-warp", "warp-lang", "tensorboard"]
        },
        "python": sys.version,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "experiment_code_sha256": {
            p.name: digest(p)
            for p in [Path(__file__), PROFILE, HERE / "monitor.py", HERE / "analyze_logs.py"]
        },
        "training_sources_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                ROOT / "src/supermimic/training/loop.py",
                ROOT / "src/supermimic/presets.py",
                ROOT / "src/supermimic/scripts/train.py",
            ]
        },
    }


def preflight(seed=0, library=None):
    out = output_root()
    out.mkdir(parents=True, exist_ok=True)
    full = make_config("full", seed, library)
    smoke = make_config("smoke", seed)
    common_errors = []
    manifest = ROOT / "data/npz/bones_seed/manifest_filtered.csv"
    full_list = ROOT / "data/bench/full.txt"
    names = []
    lengths = {}
    if not manifest.is_file() or not full_list.is_file():
        common_errors.append("Missing filtered manifest or full benchmark list")
    else:
        if digest(manifest, "md5") != EXPECTED_MANIFEST_MD5:
            common_errors.append("Filtered manifest MD5 differs from the community reference")
        names = full_list.read_text().splitlines()
        with manifest.open() as f:
            lengths = {r["name"]: int(r["frames"]) for r in csv.DictReader(f) if r["keep"] == "1"}
        if len(names) != 129491 or set(names) != set(lengths):
            common_errors.append("Filtered names/count differ from full list")
    smoke_paths = [ROOT / smoke.data.npz_dir / f"{n}.npz" for n in names[: smoke.data.limit]]
    if len(smoke_paths) != 4 or any(not p.is_file() for p in smoke_paths):
        common_errors.append("Smoke NPZ inputs missing")
    library_path = Path(full.data.library)
    if not library_path.is_absolute():
        library_path = ROOT / library_path
    library_report = inspect_library(library_path, names, lengths)
    resources = runtime_info()
    free = shutil.disk_usage(out).free
    if not resources["gpus"]:
        common_errors.append("No visible GPU reported by nvidia-smi")
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "training_started": False,
        "pretrained_checkpoint_used": False,
        "library": library_report,
        "runtime": resources,
        "disk_free_bytes": free,
        "common_errors": common_errors,
        "smoke_data_ready": not common_errors,
        "full_data_ready": not common_errors and library_report["ready"],
        "full_command": full_command(full),
        "resource_notes": {
            "full_gpu_reference": "One RTX 4090 24 GiB, community reports ~10.5 hours; actual time/memory not measured here.",
            "recommended_free_disk_gib": 30,
            "full_checkpoints_estimate": "22 files (21 numbered plus latest), roughly 5-8 GiB including optimizer/sampler states; estimate only.",
            "rebuild_library_additional_gib": 6.3,
        },
    }
    save_json(out / "preflight.json", report)
    save_json(out / "full_plan.json", asdict(full))
    save_json(out / "smoke_plan.json", asdict(smoke))
    return report


def smoke_passed(seed):
    p = output_root() / f"smoke_seed{seed}" / "smoke_checks.json"
    return p.is_file() and json.loads(p.read_text()).get("passed") is True


def verify_smoke(trainer, before, checkpoint):
    import torch

    checks = {}
    optimizers = {
        "wm": trainer.wm_trainer.opt,
        "policy": trainer.policy_trainer.opt,
        "critic": trainer.policy_trainer.critic_opt,
    }
    for name in ["wm", "policy", "critic"]:
        model = getattr(trainer, name)
        deltas = [
            float((p.detach().cpu() - before[name][k]).abs().max())
            for k, p in model.named_parameters()
        ]
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        grad_finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        steps = [int(v["step"].item()) for v in optimizers[name].state.values() if "step" in v]
        checks[name] = {
            "max_parameter_change": max(deltas),
            "parameters_finite": finite,
            "gradients_present_and_finite": grad_finite,
            "max_optimizer_step": max(steps, default=0),
        }
        if not finite or not grad_finite or max(deltas) <= 0 or not steps:
            raise RuntimeError(
                f"Smoke did not verify parameter/gradient/optimizer update for {name}"
            )
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck["iteration"] != 12 or ck["num_updates"] != 8:
        raise RuntimeError("Unexpected smoke update counts")
    if ck["config"]["resume"] is not None or ck["config"]["init_from"] is not None:
        raise RuntimeError("Smoke unexpectedly initialized from a checkpoint")
    for name in ["wm", "policy", "critic"]:
        for key, param in getattr(trainer, name).named_parameters():
            if not torch.equal(ck[name][key], param.detach().cpu()):
                raise RuntimeError("Saved parameters differ from trained parameters")
    for key in ["wm_opt", "policy_opt", "critic_opt"]:
        if not ck[key]["state"]:
            raise RuntimeError(f"Checkpoint missing optimizer state {key}")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(checkpoint.parent / "tb"), size_guidance={"scalars": 0})
    events.Reload()
    loss_values = {}
    for tag in ["wm/loss", "policy/critic_loss", "policy/imag_objective"]:
        values = events.Scalars(tag)
        if not values or not all(math.isfinite(e.value) for e in values):
            raise RuntimeError(f"Missing or nonfinite smoke forward metric: {tag}")
        loss_values[tag] = values[-1].value
    return {
        "passed": True,
        "finite_forward_metrics": loss_values,
        "iteration": ck["iteration"],
        "policy_updates": ck["num_updates"],
        "checkpoint_saved_and_reloaded": str(checkpoint.relative_to(ROOT)),
        "initialization": "new Trainer with resume=None/init_from=None; parameter snapshots taken before run()",
        "checks": checks,
    }


def run_training(mode, seed, library=None):
    cfg = make_config(mode, seed, library)
    report = preflight(seed, library)
    if not report["smoke_data_ready"]:
        raise RuntimeError("; ".join(report["common_errors"]))
    if mode == "full":
        if not report["full_data_ready"]:
            raise RuntimeError(report["library"].get("error", "Motion Library validation failed"))
        if not smoke_passed(seed):
            raise RuntimeError("Run and pass the Training Smoke Test first")
        if report["disk_free_bytes"] < 20 * 1024**3:
            raise RuntimeError("Need at least 20 GiB free for a full training run")
    gpu = next((g for g in report["runtime"]["gpus"] if g["index"] == 0), None)
    minimum = 22000 if mode == "full" else 4096
    if gpu is None or gpu["free_mib"] < minimum:
        raise RuntimeError(f"Need at least {minimum} MiB free on cuda:0")
    target = ROOT / cfg.log_root / cfg.run_name
    target.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    metadata = {
        "mode": mode,
        "seed": seed,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "runtime": report["runtime"],
        "pretrained_checkpoint_used": False,
        "resume": None,
        "resumed": False,
        "interrupted": False,
        "init_from": None,
        "native_equivalent_command": full_command(cfg) if mode == "full" else None,
        "logging": {"event_flush_seconds": 15, "gpu_sample_interval_seconds": 30},
        "command": monitored_command(cfg)
        if mode == "full"
        else "community Trainer(release + profiles.json smoke overrides).run()",
        "library": report["library"]
        if mode == "full"
        else "Four filtered BONES-SEED NPZ clips; packed library unused",
    }
    save_json(target / "run_metadata.json", metadata)
    save_json(target / "requested_config.json", asdict(cfg))
    try:
        with (target / "console.log").open("w", buffering=1) as log:
            if mode == "full":
                return_code = run_child(monitored_command(cfg), log)
                if return_code:
                    raise RuntimeError(f"Training exited {return_code}; see {target}/console.log")
            else:
                from monitor import TrainingMonitor

                with (
                    contextlib.redirect_stdout(log),
                    contextlib.redirect_stderr(log),
                    TrainingMonitor(target),
                ):
                    from supermimic.training.loop import Trainer

                    trainer = Trainer(cfg)
                    try:
                        before = {
                            name: {
                                k: p.detach().cpu().clone()
                                for k, p in getattr(trainer, name).named_parameters()
                            }
                            for name in ["wm", "policy", "critic"]
                        }
                        if trainer.iteration != 0 or trainer.num_updates != 0:
                            raise RuntimeError("Smoke did not start from iteration zero")
                        trainer.run()
                        checks = verify_smoke(trainer, before, target / "ckpt_latest.pt")
                        save_json(target / "smoke_checks.json", checks)
                    finally:
                        if trainer.writer is not None:
                            trainer.writer.close()
                        trainer.env.close()
        metadata["status"] = "completed"
    except BaseException as exc:
        child_state_path = target / "monitor_status.json"
        child_state = json.loads(child_state_path.read_text()) if child_state_path.is_file() else {}
        interrupted = (
            isinstance(exc, (KeyboardInterrupt, SystemExit))
            or child_state.get("status") == "interrupted"
        )
        metadata.update(
            status="interrupted" if interrupted else "failed",
            interrupted=interrupted,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        metadata.update(
            finished_at_utc=datetime.now(UTC).isoformat(), wall_seconds=time.monotonic() - start
        )
        save_json(target / "run_metadata.json", metadata)
        # Analysis failure must not overwrite a training failure or alter a checkpoint.
        try:
            from analyze_logs import summarize

            summarize(target)
        except Exception as exc:  # noqa: BLE001 - never mask a training failure with analysis errors
            metadata["analysis_error"] = f"{type(exc).__name__}: {exc}"
            save_json(target / "run_metadata.json", metadata)
            print(
                f"[WARN] Training log analysis failed: {type(exc).__name__}; rerun metrics.",
                file=sys.stderr,
            )
    print(f"{mode} completed: {target}")


def export_metrics(run_name, smooth_window=5):
    from analyze_logs import summarize

    run = output_root() / run_name
    if not run.resolve().is_relative_to(output_root().resolve()):
        raise ValueError("Invalid run directory")
    result = summarize(run, smoothing=smooth_window)
    print(json.dumps({"status": result["status"], "figures": result["figures"]}, indent=2))


def monitored_command(cfg):
    return [
        str(ROOT / ".venv/bin/python"),
        str(HERE / "monitor.py"),
        "--run-dir",
        str(ROOT / cfg.log_root / cfg.run_name),
        "--",
        *full_command(cfg)[1:],
    ]


def run_child(command, log):
    """Forward termination to our training child and let its event writer flush."""
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    old_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        child.terminate()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        return child.wait()
    except (KeyboardInterrupt, SystemExit):
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def rebuild_library():
    destination = output_root() / "prepared_data/bones_paper.pt"
    if destination.resolve() == (
        ROOT / DEFAULT_LIBRARY
    ).resolve() or not destination.resolve().is_relative_to(output_root().resolve()):
        raise ValueError("Rebuild must use a new path inside Experiment 007")
    if destination.exists():
        raise FileExistsError(
            "Rebuilt library already exists; run check --library instead of overwriting it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ROOT / ".venv/bin/python"),
        "scripts/data/build_library.py",
        "--npz-dir",
        "data/npz/bones_seed",
        "--manifest",
        "data/npz/bones_seed/manifest_filtered.csv",
        "--out",
        str(destination),
        "--workers",
        "8",
    ]
    metadata = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "original_library_untouched": str(DEFAULT_LIBRARY),
    }
    start = time.monotonic()
    try:
        with (output_root() / "library_build.log").open("w") as log:
            code = run_child(command, log)
        if code:
            raise RuntimeError(f"Library builder exited {code}; see library_build.log")
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata.update(status="failed", error=type(exc).__name__)
        raise
    finally:
        metadata.update(
            finished_at_utc=datetime.now(UTC).isoformat(), wall_seconds=time.monotonic() - start
        )
        save_json(output_root() / "library_build.json", metadata)
    print(f"Rebuilt library: {destination}; run check --library to validate it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["check", "rebuild-library", "smoke", "full", "metrics"],
        nargs="?",
        default="check",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--library", type=Path, default=None)
    parser.add_argument("--run-name", default="full_seed0", help="For metrics export only")
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.mode == "check":
        report = preflight(args.seed, args.library)
        print(json.dumps(report, indent=2, default=str))
        sys.exit(0 if report["full_data_ready"] else 2)
    elif args.mode == "rebuild-library":
        rebuild_library()
    elif args.mode == "metrics":
        export_metrics(args.run_name, args.smooth_window)
    else:
        run_training(args.mode, args.seed, args.library)


if __name__ == "__main__":
    main()
