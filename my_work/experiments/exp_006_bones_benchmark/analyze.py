"""BONES-SEED preflight, isolated community benchmark runner and result analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "my_work/outputs/exp_006_bones_benchmark"
sys.path.insert(0, str(ROOT / "src"))
from realmimic.eval.motion_groups import HARD_SET_GROUPS, OTHER, TABLE_A8_GROUPS, motion_group

DATA = "data/npz/bones_seed"
CHECKPOINT = "weights/realmimic_g1_86k.pt"
REFERENCE = {"clips": 129491, "mean_norm_return": 0.9545, "failed": 10}
LISTS = {
    "rand2000": (2000, "28dfbc14f02146cc05fe62c794d89f01"),
    "hard": (130, "b931bb29b8c786bfdfa34b57c0832caf"),
    "full": (129491, "e588d04ab59d948b86347088dff357f0"),
}
MANIFEST_MD5 = "5bc2aae2efc6f986ab4cf7adbb4d8ffc"
GROUP_REF = {
    "side_rolls": 0,
    "flips": 0,
    "fire_in_the_hole_turns": 0,
    "crossed_legs": 2,
    "lying_resting_standing_up": 3,
    "broad_jumps": 0,
    "dance_routine": 0,
    OTHER: 5,
}


def now():
    return datetime.now(UTC).isoformat()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def checksum(path, algorithm="md5"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read_list(path):
    names = path.read_text().splitlines()
    if not names or any(not n.strip() or n != n.strip() or "/" in n or "\\" in n for n in names):
        raise ValueError(f"Invalid motion list {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate motion names in {path}")
    return names


def gpu_info():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("nvidia-smi failed: " + result.stderr.strip())
    rows = []
    for line in result.stdout.splitlines():
        index, name, total, used, free, util = [x.strip() for x in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "total_mib": int(total),
                "used_mib": int(used),
                "free_mib": int(free),
                "utilization_percent": int(util),
            }
        )
    return rows


def active_preprocessing():
    result = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, check=True)
    targets = (
        "tools/prepare_data.sh",
        "scripts/data/csv_to_npz_batch.py",
        "scripts/data/filter_bones_seed.py",
        "scripts/data/build_library.py",
        "tools/make_benchmark_sets.py",
    )
    return [
        int(line.split(None, 1)[0])
        for line in result.stdout.splitlines()[1:]
        if any(target in line for target in targets)
        and not any(shell in line for shell in ["bash -c", "python - <<"])
    ]


def validate_sample_npz(path):
    import numpy as np

    with zipfile.ZipFile(path) as archive:
        expected = [
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "body_names",
            "joint_names",
        ]
        for key in expected:
            if key + ".npy" not in archive.namelist():
                raise ValueError(f"{path}: missing {key}")
        shapes = {}
        for key in [
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ]:
            with archive.open(key + ".npy") as member:
                version = np.lib.format.read_magic(member)
                reader = {
                    (1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0,
                }.get(version)
                if reader is None:
                    raise ValueError(f"Unsupported NPY header version {version}")
                shape, _, _ = reader(member)
                shapes[key] = shape
    frames = shapes["joint_pos"][0]
    if frames < 2 or shapes["joint_pos"] != (frames, 29) or shapes["joint_vel"] != (frames, 29):
        raise ValueError(f"{path}: invalid joint dimensions")
    for key, width in [
        ("body_pos_w", 3),
        ("body_quat_w", 4),
        ("body_lin_vel_w", 3),
        ("body_ang_vel_w", 3),
    ]:
        if len(shapes[key]) != 3 or shapes[key][0] != frames or shapes[key][2] != width:
            raise ValueError(f"{path}: invalid body dimensions")
    with np.load(path, allow_pickle=False) as data:
        if not np.isclose(float(data["fps"].reshape(-1)[0]), 50):
            raise ValueError(f"{path}: fps is not 50")


def preflight():
    errors = []
    sets = {}
    details = {}
    pids = active_preprocessing()
    if pids:
        errors.append(f"Preprocessing still active (PIDs {pids}); evaluation blocked")
    manifest = ROOT / DATA / "manifest_filtered.csv"
    kept = set()
    if not manifest.is_file():
        errors.append("Missing filtered manifest; preprocessing incomplete")
    else:
        md5 = checksum(manifest)
        details["manifest_md5"] = md5
        if md5 != MANIFEST_MD5:
            errors.append("Filtered manifest MD5 differs from community reference")
        with manifest.open() as f:
            kept = {r["name"] for r in csv.DictReader(f) if r.get("keep") == "1"}
        details["kept_clips"] = len(kept)
        if len(kept) != 129491:
            errors.append(f"Expected 129491 kept clips, found {len(kept)}")
    for stage, (expected, expected_md5) in LISTS.items():
        path = ROOT / f"data/bench/{stage}.txt"
        if not path.is_file():
            errors.append(f"Missing {path.relative_to(ROOT)}; preprocessing incomplete")
            continue
        try:
            names = read_list(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        md5 = checksum(path)
        sets[stage] = names
        details[stage] = {
            "clips": len(names),
            "md5": md5,
            "expected_clips": expected,
            "expected_md5": expected_md5,
        }
        if len(names) != expected or md5 != expected_md5:
            errors.append(f"{stage}: count or MD5 differs from reference")
    if "full" in sets and set(sets["full"]) != kept:
        errors.append("Full list differs from kept manifest names")
    if "rand2000" in sets and not set(sets["rand2000"]) <= kept:
        errors.append("Random list contains unfiltered clips")
    if "hard" in sets and set(sets["hard"]) != {
        n for n in kept if motion_group(n) in HARD_SET_GROUPS
    }:
        errors.append("Hard list differs from community HARD_SET_GROUPS")
    directory = ROOT / DATA
    if not directory.is_dir():
        errors.append("Missing converted BONES-SEED directory")
    elif sets:
        files = {
            e.name[:-4] for e in os.scandir(directory) if e.name.endswith(".npz") and e.is_file()
        }
        details["available_npz_files"] = len(files)
        missing = kept - files
        if missing:
            errors.append(f"{len(missing)} filtered clips have no NPZ file")
        # Header/FPS validation is sampled; full array checks happen in the community loader.
        full = sets.get("full", [])
        sampled = sorted({full[i * (len(full) - 1) // 15] for i in range(16)}) if full else []
        for name in sampled:
            try:
                validate_sample_npz(directory / f"{name}.npz")
            except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as exc:
                errors.append(f"Invalid sampled NPZ: {type(exc).__name__}")
        details["npz_header_fps_samples_checked"] = len(sampled)
    if not (ROOT / CHECKPOINT).is_file():
        errors.append("Missing community checkpoint")
    try:
        gpus = gpu_info()
    except (RuntimeError, FileNotFoundError) as exc:
        gpus = []
        errors.append(str(exc))
    report = {
        "checked_at_utc": now(),
        "ready": not errors,
        "errors": errors,
        "data": details,
        "gpu": gpus,
        "checkpoint": CHECKPOINT,
        "reference": REFERENCE,
        "npz_validation_scope": "All filtered names must exist; 16 distributed NPZ headers/FPS checked; data arrays validated by community loader at evaluation.",
        "full_benchmark_started": False,
    }
    save_json(OUT / "preflight.json", report)
    return report


def summarize_result(data, expected_names):
    clips = data["clips"]
    names = [c["name"] for c in clips]
    if len(names) != len(set(names)) or set(names) != set(expected_names):
        raise ValueError("Result motion names/count differ from the requested list")
    if not clips:
        raise ValueError("Empty benchmark result")
    count = len(clips)
    frames = sum(c["frames"] for c in clips)
    for c in clips:
        if type(c["failed"]) is not bool or c["frames"] < 1 or not 0 < c["steps"] <= c["frames"]:
            raise ValueError("Invalid failure flags or step counts")
        if c["fail_step"] != (c["steps"] if c["failed"] else -1):
            raise ValueError("Invalid failure timing")
        if not c["failed"] and c["steps"] != c["frames"]:
            raise ValueError("Incomplete successful clip")
        if not math.isfinite(c["norm_return"]):
            raise ValueError("Nonfinite tracking score")
    result = {
        "clips": count,
        "failed": sum(c["failed"] for c in clips),
        "mean_norm_return": sum(c["norm_return"] for c in clips) / count,
        "step_weighted_norm_return": sum(c["norm_return"] * c["frames"] for c in clips) / frames,
        "failed_at_end": sum(c["failed"] and c["fail_step"] >= c["frames"] - 3 for c in clips),
    }
    for key, value in result.items():
        if not math.isclose(value, data["summary"][key], abs_tol=1e-9):
            raise ValueError(f"Stored summary mismatch: {key}")
    result["failure_rate"] = result["failed"] / count
    result["overflow_clips"] = sum(c.get("overflow_steps", 0) > 0 for c in clips)
    result["overflow_steps"] = sum(c.get("overflow_steps", 0) for c in clips)
    return result


def run_stage(stage, args):
    output = OUT / f"{stage}.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; refusing to overwrite. Use analyze.")
    report = preflight()
    if not report["ready"]:
        raise RuntimeError("; ".join(report["errors"]))
    index = int(args.device.split(":")[1])
    gpu = next((g for g in report["gpu"] if g["index"] == index), None)
    if gpu is None or gpu["free_mib"] < 6144:
        raise RuntimeError("Selected GPU missing or less than 6 GiB free; evaluation not started")
    if stage == "full" and any(not (OUT / f"{s}.json").is_file() for s in ["rand2000", "hard"]):
        raise RuntimeError("Complete and inspect rand2000/hard before explicitly running full")
    checkpoint_hash = checksum(ROOT / CHECKPOINT, "sha256")
    for prior in OUT.glob("*_run.json"):
        previous = json.loads(prior.read_text())
        if previous["status"] == "completed" and previous["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Checkpoint differs from an earlier completed stage")
    lock = OUT / ".evaluation.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    command = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        "scripts/eval_benchmark.py",
        "--ckpt",
        CHECKPOINT,
        "--npz-dir",
        DATA,
        "--clip-list",
        f"data/bench/{stage}.txt",
        "--num-envs",
        str(args.num_envs),
        "--chunk-size",
        str(args.chunk_size),
        "--device",
        args.device,
        "--out",
        str(output.relative_to(ROOT)),
    ]
    metadata = {
        "stage": stage,
        "status": "running",
        "started_at_utc": now(),
        "command": command,
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": checkpoint_hash,
        "list_path": f"data/bench/{stage}.txt",
        "list_md5": report["data"][stage]["md5"],
        "manifest_md5": report["data"]["manifest_md5"],
        "num_envs_requested": args.num_envs,
        "chunk_size": args.chunk_size,
        "effective_envs_per_chunk": [
            min(args.num_envs, n)
            for n in [
                min(args.chunk_size, LISTS[stage][0] - i)
                for i in range(0, LISTS[stage][0], args.chunk_size)
            ]
        ],
        "device": args.device,
        "gpu_before": gpu,
        "python": sys.version,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["torch", "mujoco", "mujoco-warp", "warp-lang", "mjlab"]
        },
        "community_evaluator_sha256": checksum(ROOT / "scripts/eval_benchmark.py", "sha256"),
        "experiment_runner_sha256": checksum(Path(__file__), "sha256"),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "training_performed": False,
        "policy_weights": "policy (not EMA)",
        "benchmark_protocol": "Community run_benchmark: deterministic start, no noise/reset noise/domain randomization, spec gains, seed 0; env_kwargs from checkpoint",
    }
    start = time.monotonic()
    try:
        save_json(OUT / f"{stage}_run.json", metadata)
        logs = OUT / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / f"{stage}.log").open("w") as log:
            env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            completed = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode:
            raise RuntimeError(f"Evaluator exited {completed.returncode}; see {logs / stage}.log")
        data = json.loads(output.read_text())
        result = summarize_result(data, read_list(ROOT / f"data/bench/{stage}.txt"))
        metadata.update(status="completed", result=result)
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata.update(finished_at_utc=now(), wall_seconds=time.monotonic() - start)
        save_json(OUT / f"{stage}_run.json", metadata)
        lock.unlink(missing_ok=True)
    analyze()
    print(json.dumps({"stage": stage, **result}, indent=2), flush=True)


def write_csv(path, rows, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    rows = []
    failures = []
    groups = []
    for stage in LISTS:
        path = OUT / f"{stage}.json"
        if not path.is_file():
            continue
        metadata_path = OUT / f"{stage}_run.json"
        if (
            not metadata_path.is_file()
            or json.loads(metadata_path.read_text())["status"] != "completed"
        ):
            raise ValueError(f"{stage}: no completed run metadata")
        data = json.loads(path.read_text())
        result = summarize_result(data, read_list(ROOT / f"data/bench/{stage}.txt"))
        metadata = json.loads(metadata_path.read_text())
        rows.append(
            {
                "stage": stage,
                **result,
                "num_envs_requested": metadata["num_envs_requested"],
                "chunk_size": metadata["chunk_size"],
                "wall_seconds": metadata["wall_seconds"],
                "comparable_to_full_reference": stage == "full",
                "score_delta_from_full_reference": result["mean_norm_return"] - 0.9545
                if stage == "full"
                else None,
                "failed_delta_from_full_reference": result["failed"] - 10
                if stage == "full"
                else None,
            }
        )
        for c in data["clips"]:
            if c["failed"]:
                failures.append(
                    {
                        "stage": stage,
                        "name": c["name"],
                        "category": motion_group(c["name"]),
                        "norm_return": c["norm_return"],
                        "frames": c["frames"],
                        "fail_step": c["fail_step"],
                        "fail_time_s": c["fail_step"] * 0.02,
                        "overflow_steps": c.get("overflow_steps", 0),
                    }
                )
        for group in [*TABLE_A8_GROUPS, OTHER]:
            cs = [c for c in data["clips"] if motion_group(c["name"]) == group]
            groups.append(
                {
                    "stage": stage,
                    "category": group,
                    "clips": len(cs),
                    "failed": sum(c["failed"] for c in cs),
                    "failure_rate": sum(c["failed"] for c in cs) / len(cs) if cs else None,
                    "mean_norm_return": sum(c["norm_return"] for c in cs) / len(cs) if cs else None,
                    "community_full_failed_reference": GROUP_REF[group]
                    if stage == "full"
                    else None,
                }
            )
    if not rows:
        print("No completed benchmark results; no scores or figures generated.")
        return
    write_csv(OUT / "summary.csv", rows, list(rows[0]))
    write_csv(
        OUT / "failures.csv",
        failures,
        [
            "stage",
            "name",
            "category",
            "norm_return",
            "frames",
            "fail_step",
            "fail_time_s",
            "overflow_steps",
        ],
    )
    write_csv(OUT / "category_summary.csv", groups, list(groups[0]))
    plot(rows, groups)
    lines = [
        "# Experiment 006: BONES-SEED Benchmark Reproduction",
        "",
        "Pretrained community policy only; no training. BONES-SEED is training-related data, not unseen-motion generalization.",
        "",
        "| Stage | Clips | Mean tracking score | Failed | Failure rate | Overflow clips |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['stage']} | {r['clips']} | {r['mean_norm_return']:.6f} | {r['failed']} | {r['failure_rate']:.4%} | {r['overflow_clips']} |"
        for r in rows
    ]
    lines += [
        "",
        "## Community reference",
        "",
        "Full benchmark: 129,491 clips, mean normalized return 0.9545, 10 failed. Random/hard subsets have different distributions; they are not direct reproductions of the full-set mean or failure count.",
    ]
    if not any(r["stage"] == "full" for r in rows):
        lines += [
            "",
            "**Full benchmark has not been run. It requires the user to inspect and confirm the first two stages.**",
        ]
    lines += [
        "",
        "## Failure analysis",
        "",
        "Local `failures.csv` contains exact failure names, timing, score and overflow flags. `category_summary.csv` uses the community Table A.8 name patterns; zero-sample categories have no rate. The 130-clip hard set covers only five HARD_SET_GROUPS, not every Table A.8 category.",
        "",
        "## Reproducibility",
        "",
        "`*_run.json` records the actual command, checkpoint hash, list hashes, requested/effective parallelism, chunk size, GPU resources, software versions and elapsed wall time. Raw JSON summaries are recomputed and exact motion coverage is checked.",
        "",
        "Main tracking score is mean norm_return, penalizing early termination. Frame-weighted return is separate. The survived-step legacy score is not substituted. Overflow diagnostics are reported and may affect physical validity; numerical/contact behavior can depend on parallelism.",
        "",
        "Raw per-clip JSON, failure names and logs remain local/ignored; aggregate tables and figures may be committed. Source data, lists and weights remain in their existing ignored directories.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def plot(rows, groups):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    bars = axes[0].bar(
        [r["stage"] for r in rows], [r["mean_norm_return"] for r in rows], color="#4479a8"
    )
    axes[0].bar_label(bars, fmt="%.4f", padding=3)
    axes[0].set(
        ylim=(0, 1.08),
        ylabel="Mean normalized tracking score",
        title="Observed scores (different clip sets)",
    )
    axes[0].axhline(
        0.9545, color="gray", linestyle="--", label="Community full-set reference: 0.9545"
    )
    axes[0].legend(fontsize=8)
    bars = axes[1].bar(
        [r["stage"] for r in rows], [100 * r["failure_rate"] for r in rows], color="#cf8c42"
    )
    axes[1].bar_label(bars, labels=[f"{r['failed']}/{r['clips']}" for r in rows], padding=3)
    axes[1].set(ylabel="Failed clips (%)", title="Failure rate; labels show failed / total")
    axes[1].set_ylim(0, max(1, max(r["failure_rate"] for r in rows) * 130))
    for ext in ["png", "pdf"]:
        fig.savefig(folder / f"benchmark_summary.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    chosen = [r for r in groups if r["stage"] == "hard" and r["clips"]]
    if chosen:
        fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
        labels = [r["category"].replace("_", " ") for r in chosen]
        bars = ax.barh(labels, [r["failed"] for r in chosen], color="#b95f54")
        ax.bar_label(bars, labels=[f"{r['failed']}/{r['clips']}" for r in chosen], padding=3)
        ax.set(
            xlabel="Failed clips", title="Challenging motion benchmark: failures / category samples"
        )
        ax.set_xlim(0, max(1, max(r["failed"] for r in chosen) + 1))
        for ext in ["png", "pdf"]:
            fig.savefig(folder / f"hard_category_failures.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["check", "rand2000", "hard", "full", "analyze"],
        nargs="?",
        default="check",
    )
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (
        args.num_envs < 1
        or args.chunk_size < 1
        or not args.device.startswith("cuda:")
        or not args.device[5:].isdigit()
    ):
        parser.error("Use positive num-envs/chunk-size and a cuda:N device")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "check":
        status = preflight()
        print(json.dumps(status, indent=2))
        sys.exit(0 if status["ready"] else 2)
    elif args.stage == "analyze":
        analyze()
    else:
        run_stage(args.stage, args)


if __name__ == "__main__":
    main()
