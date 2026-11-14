"""Analyze real native TensorBoard/console data; never run training or invent absent curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_ROOT = ROOT / "my_work/outputs/exp_007_full_training"
PLOTS = [
    ("01_world_model_loss", "wm/loss", "World Model Training Loss", "Training loss"),
    (
        "02_world_model_prediction_metric",
        "wm/online_nse_8step",
        "World Model Prediction Metric",
        "Online 8-step normalized squared error",
    ),
    (
        "03_policy_imagination_objective",
        "policy/imag_objective",
        "Policy Imagination Objective",
        "Imagination objective (higher is better)",
    ),
    ("04_critic_loss", "policy/critic_loss", "Critic Training Loss", "Critic loss"),
    (
        "05_training_throughput",
        "derived/iterations_per_second",
        "Training Throughput",
        "Iterations per second",
    ),
]
THROUGHPUT = re.compile(r"^it (\d+) \| .*? \| ([\d.]+) it/s \| env steps (\d+)")


def load_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def scalar_rows(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    rows = []
    # Per-file, no orphan purging: preserve restart/overlap history, including nonfinite values.
    for path in sorted((run / "tb").glob("events.out.tfevents.*")):
        events = EventAccumulator(
            str(path), size_guidance={"scalars": 0}, purge_orphaned_data=False
        )
        events.Reload()
        for tag in events.Tags()["scalars"]:
            rows.extend(
                {
                    "event_file": path.name,
                    "tag": tag,
                    "step": e.step,
                    "wall_time": e.wall_time,
                    "value": e.value,
                }
                for e in events.Scalars(tag)
            )
    return rows


def throughput_rows(run, scalars):
    rows = []
    log = run / "console.log"
    if log.is_file():
        with log.open(errors="replace") as f:
            segment = 0
            previous = -1
            for line in f:
                match = THROUGHPUT.match(line)
                if not match:
                    continue
                step, speed, env_steps = int(match[1]), float(match[2]), int(match[3])
                if step <= previous:
                    segment += 1
                rows.append(
                    {
                        "event_file": f"console_segment_{segment}",
                        "tag": "derived/iterations_per_second",
                        "step": step,
                        "wall_time": None,
                        "value": speed,
                        "source": "community console it/s",
                        "env_steps": env_steps,
                    }
                )
                previous = step
    if rows:
        return rows
    previous = None
    for r in sorted((r for r in scalars if r["tag"] == "time/hours"), key=lambda r: r["wall_time"]):
        if previous is not None and r["event_file"] == previous["event_file"]:
            ds, dt = r["step"] - previous["step"], (r["value"] - previous["value"]) * 3600
            if ds > 0 and math.isfinite(dt) and dt > 0:
                rows.append(
                    {
                        **r,
                        "tag": "derived/iterations_per_second",
                        "value": ds / dt,
                        "source": "delta(step)/delta(native time/hours)",
                        "env_steps": None,
                    }
                )
        previous = r
    return rows


def trailing_mean(values, window):
    """Trailing mean of at most N finite logged points; restart after nonfinite samples."""
    buffer = []
    result = []
    for value in values:
        if not math.isfinite(value):
            buffer.clear()
            result.append(float("nan"))
        else:
            buffer.append(value)
            buffer = buffer[-window:]
            result.append(sum(buffer) / len(buffer))
    return result


def draw(rows, folder, boundary, smoothing):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 18})
    folder.mkdir(exist_ok=True)
    paths = []
    for filename, tag, title, ylabel in PLOTS:
        data = sorted(
            (r for r in rows if r["tag"] == tag),
            key=lambda r: (r["event_file"], r["wall_time"] or 0, r["step"]),
        )
        if not data or not any(math.isfinite(r["value"]) for r in data):
            continue
        fig, ax = plt.subplots(figsize=(12.8, 7.2), layout="constrained")
        segments = []
        for row in data:
            if (
                not segments
                or row["event_file"] != segments[-1][-1]["event_file"]
                or row["step"] <= segments[-1][-1]["step"]
            ):
                segments.append([])
            segments[-1].append(row)
        for index, segment in enumerate(segments):
            x, y = [r["step"] for r in segment], [r["value"] for r in segment]
            ax.plot(
                x,
                y,
                color="#4479a8",
                alpha=0.35,
                linewidth=1,
                label="Raw logged values" if index == 0 else None,
            )
            ax.plot(
                x,
                trailing_mean(y, smoothing),
                color="#164c80",
                linewidth=2,
                label=f"Trailing mean: up to {smoothing} logged points" if index == 0 else None,
            )
        maximum = max(max(r["step"] for r in data), boundary) * 1.03
        ax.axvspan(0, boundary, color="#e2ebf4", alpha=0.5, label="World Model-only phase")
        ax.axvline(
            boundary, color="#a65b32", linestyle="--", label=f"Joint-training boundary: {boundary}"
        )
        ax.set(xlim=(0, maximum), xlabel="Training iteration", ylabel=ylabel, title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=10)
        for ext in ["png", "pdf"]:
            path = folder / f"{filename}.{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            paths.append(str(path.name))
        plt.close(fig)
    return paths


def checkpoint_info(path):
    if not path.is_file():
        return {}
    import torch

    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        updates = {}
        for name in ["wm_opt", "policy_opt", "critic_opt"]:
            steps = [
                int(s["step"].item())
                for s in ck.get(name, {}).get("state", {}).values()
                if "step" in s
            ]
            updates[name] = max(steps) if steps else None
        nonfinite = []
        for name in ["wm", "policy", "critic"]:
            for key, value in ck.get(name, {}).items():
                if value.is_floating_point() and not torch.isfinite(value).all():
                    nonfinite.append(f"{name}.{key}")
        return {
            "path": str(path.name),
            "iteration": ck["iteration"],
            "env_steps": ck["env_steps"],
            "policy_updates": ck.get("num_updates"),
            "train_seconds": ck.get("train_seconds"),
            "optimizer_steps": updates,
            "nonfinite_model_tensors": nonfinite,
            "resume": str(ck["config"]["resume"]) if ck["config"].get("resume") else None,
            "init_from": str(ck["config"]["init_from"]) if ck["config"].get("init_from") else None,
        }
    except (RuntimeError, OSError, ValueError, KeyError, EOFError) as exc:
        return {"path": str(path.name), "read_error": type(exc).__name__}


def summarize(run, smoothing=5, inspect_checkpoint=True):
    run = Path(run)
    if smoothing < 1:
        raise ValueError("Smoothing window must be positive")
    if not run.is_dir():
        # Do not create full_seed0: fresh training deliberately refuses existing run directories.
        return {
            "status": "not_started",
            "figures": [],
            "message": "No run directory; no output created.",
        }
    metadata = load_json(run / "run_metadata.json")
    monitor = load_json(run / "monitor_status.json")
    cfg = load_json(run / "config.json") or load_json(run / "requested_config.json")
    boundary = cfg.get("schedule", {}).get("wm_warmup_iterations", 500)
    expected = cfg.get("schedule", {}).get("total_iterations")
    scalars = scalar_rows(run)
    throughput = throughput_rows(run, scalars)
    by_tag = defaultdict(list)
    for row in scalars:
        by_tag[row["tag"]].append(row)
    metrics = {}
    nonfinite_count = 0
    for tag, values in by_tag.items():
        values.sort(key=lambda r: r["wall_time"])
        finite = [r for r in values if math.isfinite(r["value"])]
        bad = len(values) - len(finite)
        nonfinite_count += bad
        metrics[tag] = {
            "samples": len(values),
            "nonfinite_samples": bad,
            "first_finite_value": finite[0]["value"] if finite else None,
            "last_finite_value": finite[-1]["value"] if finite else None,
            "last_step": values[-1]["step"],
        }
    status = metadata.get("status", monitor.get("status", "not_started"))
    process_alive = None
    if monitor.get("pid"):
        try:
            os.kill(monitor["pid"], 0)
            process_alive = True
        except ProcessLookupError:
            process_alive = False
        except PermissionError:
            process_alive = True
    abrupt_end = monitor.get("status") == "running" and process_alive is False
    if status == "running" and process_alive is False:
        status = "interrupted_or_unfinished"
    final = (
        checkpoint_info(run / "ckpt_latest.pt")
        if inspect_checkpoint and status != "running"
        else {}
    )
    complete = status == "completed" and expected is not None and final.get("iteration") == expected
    target_reached = expected is not None and final.get("iteration") == expected
    observed = [r["step"] for r in scalars]
    anomalies = []
    if nonfinite_count:
        anomalies.append(f"{nonfinite_count} nonfinite scalar samples; raw CSV preserves them.")
    if final.get("nonfinite_model_tensors"):
        anomalies.append("Final checkpoint contains nonfinite model tensors.")
    if monitor.get("gpu_sampling_error"):
        anomalies.append(f"Process GPU sampling unavailable: {monitor['gpu_sampling_error']}")
    if final.get("read_error"):
        anomalies.append("Latest checkpoint is not readable; it may be incomplete.")
    for tag, rows in by_tag.items():
        if (tag.startswith(("sim/overflow_", "sampler/unphysical_"))) and any(
            r["value"] > 0 for r in rows if math.isfinite(r["value"])
        ):
            anomalies.append(f"Positive diagnostic recorded: {tag}")
    restarts = any(
        any(b["step"] <= a["step"] for a, b in pairwise(rows)) for rows in by_tag.values()
    )
    if abrupt_end:
        anomalies.append(
            "Training process ended without monitor finalization; forced termination is possible."
        )
    if restarts:
        anomalies.append("Repeated/decreasing steps detected; raw restart history retained.")
    if status in ["failed", "interrupted", "interrupted_or_unfinished"]:
        anomalies.append(f"Run status: {status}")
    summaries = {
        "status": status,
        "profile": metadata.get("mode"),
        "training_completed_normally": complete
        and not nonfinite_count
        and not final.get("nonfinite_model_tensors"),
        "reached_target_iterations": target_reached,
        "training_started": bool(scalars) or bool(metadata),
        "started_at_utc": metadata.get("started_at_utc", monitor.get("started_at_utc")),
        "finished_at_utc": metadata.get("finished_at_utc", monitor.get("finished_at_utc")),
        "total_process_wall_seconds": metadata.get(
            "wall_seconds", monitor.get("process_wall_seconds")
        ),
        "training_elapsed_seconds": final.get("train_seconds"),
        "target_iterations": expected,
        "actual_iterations": final.get("iteration") if target_reached else None,
        "actual_environment_steps": final.get("env_steps") if target_reached else None,
        "last_logged_iteration": max(observed) if observed else None,
        "last_logged_environment_steps": metrics.get("time/env_steps", {}).get("last_finite_value"),
        "last_logged_policy_updates": metrics.get("time/policy_updates", {}).get(
            "last_finite_value"
        ),
        "checkpoint": final or None,
        "runtime": metadata.get("runtime"),
        "seed": cfg.get("seed", metadata.get("seed")),
        "started_from_random_initialization": metadata.get("pretrained_checkpoint_used") is False
        and metadata.get("resume") is None
        and metadata.get("init_from") is None,
        "interrupted": status in ["interrupted", "interrupted_or_unfinished"] or abrupt_end,
        "resumed": bool(cfg.get("resume")) or bool(final.get("resume")),
        "step_restart_detected": restarts,
        "training_config": cfg or None,
        "gpu_memory": {
            key: monitor.get(key)
            for key in [
                "gpu_torch_peak_allocated_mib",
                "gpu_torch_peak_reserved_mib",
                "gpu_sampled_process_peak_mib",
            ]
        },
        "gpu_memory_scope": "PyTorch counters exclude Warp; process peak is sampled every 30 s and may miss brief spikes.",
        "metric_summary": metrics,
        "anomalies": anomalies,
        "world_model_only_boundary": boundary,
        "smoothing": f"Trailing arithmetic mean of up to {smoothing} logged points; reset at event file/step restarts and nonfinite values; no interpolation.",
        "throughput_source": sorted({r["source"] for r in throughput}),
        "figures": [],
    }
    if scalars:
        folder = run / "metrics"
        folder.mkdir(exist_ok=True)
        for filename, rows, columns in [
            ("scalars.csv", scalars, ["event_file", "tag", "step", "wall_time", "value"]),
            (
                "throughput.csv",
                throughput,
                ["event_file", "tag", "step", "wall_time", "value", "source", "env_steps"],
            ),
        ]:
            with (folder / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
        summaries["figures"] = draw([*scalars, *throughput], run / "figures", boundary, smoothing)
    from monitor import atomic_json

    atomic_json(run / "training_summary.json", summaries)
    lines = [
        "# Training Report",
        "",
        f"Run status: **{status}**. Normal completion verified: **{summaries['training_completed_normally']}**.",
        "",
        "## Training stages",
        "",
        f"World Model-only updates through iteration {boundary}; subsequent iterations update World Model, Policy and Critic. Initial data collection precedes these updates.",
        "",
        "## Progress",
        "",
        f"- Target iterations: {expected}.",
        f"- Verified final completed iterations: {summaries['actual_iterations']}.",
        f"- Verified final environment steps: {summaries['actual_environment_steps']}.",
        f"- Last logged iteration: {summaries['last_logged_iteration']} (a lower bound after interruption).",
        f"- Total process wall time (seconds, including startup): {summaries['total_process_wall_seconds']}.",
        f"- Latest checkpoint: {final.get('path', 'not available')}; checkpoint iteration: {final.get('iteration')}.",
        "",
        "## Metric changes",
        "",
        "| Metric | First finite value | Last finite value | Nonfinite samples |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {tag} | {metrics[tag]['first_finite_value']} | {metrics[tag]['last_finite_value']} | {metrics[tag]['nonfinite_samples']} |"
        for tag in [
            "wm/loss",
            "wm/state_loss",
            "wm/term_bce",
            "wm/online_nse_8step",
            "policy/imag_objective",
            "policy/critic_loss",
        ]
        if tag in metrics
    ]
    lines += [
        "",
        "Loss, imagination objective and online prediction metric are different quantities. The online prediction metric is measured on training replay, not a held-out Benchmark.",
        "",
        "## Logging and plotting",
        "",
        summaries["smoothing"],
        "Raw logged samples, including nonfinite and duplicate/restart points, remain in metrics/scalars.csv. Existing console it/s is reused; TensorBoard time differences are a fallback. Five separate PNG/PDF curves are generated only for available data.",
        "",
        "## Memory and stability",
        "",
        summaries["gpu_memory_scope"],
        f"GPU memory (MiB): {summaries['gpu_memory']}.",
        "",
        *(
            ["- " + issue for issue in anomalies]
            if anomalies
            else [
                "No anomaly was found in the available records; this does not prove every unlogged step was finite."
            ]
        ),
        "",
        "## Provenance",
        "",
        "training_summary.json includes complete configuration, seed, runtime versions, source hashes, commit and initialization/resume flags when available. A missing value means unmeasured, not zero.",
        "",
        "Logs flush at 15-second intervals without adding evaluations. SIGKILL, power loss and failures before a log point can still lose unflushed or not-yet-emitted data; no checkpoint is created by this analyzer.",
    ]
    (run / "TRAINING_REPORT.md").write_text("\n".join(lines) + "\n")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_seed0")
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()
    run = DEFAULT_ROOT / args.run_name
    if not run.resolve().is_relative_to(DEFAULT_ROOT.resolve()):
        parser.error("Run path escapes Experiment 007")
    summary = summarize(run, args.smooth_window)
    print(json.dumps({"status": summary["status"], "figures": summary["figures"]}, indent=2))


if __name__ == "__main__":
    main()
