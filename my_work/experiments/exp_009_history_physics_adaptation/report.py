"""Plot measured training/evaluation data only; missing experiments remain explicitly missing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT, output_path, save_json


def figure(fig, path):
    for ext in ["png", "pdf"]:
        fig.savefig(path.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=OUT / "validation")
    p.add_argument("--eval-dir", type=Path)
    args = p.parse_args()
    run = output_path(args.run_dir)
    if not run.exists():
        raise FileNotFoundError(run)
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    made = []
    missing = []
    check = (
        json.loads((run / "full_check.json").read_text())
        if (run / "full_check.json").exists()
        else None
    )
    if check is not None:
        calibration = check.get("calibration")
        if calibration:
            phases = ["initial_zero_adapter", "temporary_nonzero_output_layer"]
            labels = ["Tracking loss |−J|", "Friction MSE", "Raw residual L2²"]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
            for i, phase in enumerate(phases):
                values = calibration[phase]
                keys = ["tracking", "friction", "raw_residual_regularization"]
                axes[0].bar(
                    np.arange(3) + (i - 0.5) * 0.35,
                    [abs(values[k]["value"]) for k in keys],
                    0.35,
                    label=["Zero output layer", "Temporary nonzero output"][i],
                )
                axes[1].bar(
                    np.arange(2) + (i - 0.5) * 0.35,
                    [values["tracking"]["gradients"][k] for k in ["encoder", "adapter"]],
                    0.35,
                    label=["Zero output layer", "Temporary nonzero output"][i],
                )
            axes[0].set(
                xticks=range(3),
                xticklabels=labels,
                yscale="symlog",
                ylabel="Absolute unweighted loss (symlog; zeros preserved)",
                title="Measured scales on real BONES-SEED starts",
            )
            axes[1].set(
                xticks=range(2),
                xticklabels=["Physics Encoder", "Residual Adapter"],
                yscale="symlog",
                ylabel="Tracking-loss gradient norm (symlog)",
                title="Expected zero-init gate; nonzero downstream gradient",
            )
            for ax in axes:
                ax.legend(fontsize=8)
            axes[0].set_yscale("symlog", linthresh=1e-6)
            axes[1].set_yscale("symlog", linthresh=1e-7)
            fig.suptitle("Correctness diagnostics only: B=8, H=4; not fine-tuning results")
            figure(fig, output / "validation_loss_and_gradients")
            made.append("validation_loss_and_gradients")
        memory = run / "gpu_process_memory.json"
        if memory.exists():
            m = json.loads(memory.read_text())
            s = m["samples"]
            if s:
                fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
                ax.plot(
                    [r["unix_time"] - s[0]["unix_time"] for r in s],
                    [r["process_used_mib"] / 1024 for r in s],
                )
                ax.set(
                    xlabel="Time since monitor attachment (s)",
                    ylabel="Observed process GPU allocation (GiB)",
                    title="One full-size validation update: 4096 envs, B=2048, H=32\nCompilation included; observed process allocation, not steady-state throughput",
                )
                figure(fig, output / "validation_gpu_memory")
                made.append("validation_gpu_memory")
    metrics = run / "metrics.csv"
    if metrics.exists():
        rows = list(csv.DictReader(metrics.open()))
        x = [int(r["iteration"]) for r in rows]
        exp = json.loads((run / "experiment.json").read_text())
        lm = exp["settings"]["lambda_mu"]
        la = exp["settings"]["lambda_action"]
        fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
        for key, label, scale in [
            ("imag_objective", "Original tracking loss −J", -1),
            ("friction_mse", "Weighted friction MSE", lm),
            ("action_regularization", "Weighted raw residual regularization", la),
            ("total_loss", "Total joint loss", 1),
        ]:
            ax.plot(x, [float(r[key]) * scale for r in rows], label=label)
        ax.set(
            xlabel="Joint fine-tuning iteration",
            ylabel="Loss (original objective scale)",
            title="Joint Training Loss Components",
        )
        ax.legend()
        figure(fig, output / "01_joint_training_loss")
        made.append("01_joint_training_loss")
        for key, title, ylabel, name in [
            (
                "imag_objective",
                "Tracking Reward During Fine-Tuning",
                "Original 32-step objective J, including critic bootstrap",
                "02_tracking_reward",
            ),
            (
                "friction_mae",
                "Friction Estimation MAE During Training",
                "Survival-weighted imagined friction MAE",
                "03_friction_mae",
            ),
            (
                "residual_abs_mean",
                "Residual Action Magnitude During Training",
                "Mean |α tanh(raw residual)|, normalized action units",
                "07_residual_magnitude",
            ),
        ]:
            fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
            ax.plot(x, [float(r[key]) for r in rows])
            ax.set(xlabel="Joint fine-tuning iteration", ylabel=ylabel, title=title)
            ax.grid(alpha=0.2)
            figure(fig, output / name)
            made.append(name)
    else:
        missing += [
            "01_joint_training_loss",
            "02_tracking_reward",
            "03_friction_mae",
            "07_residual_magnitude",
        ]
    predictions = run / "friction_predictions.jsonl"
    if predictions.exists():
        rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        pts = ax.scatter(
            [r["ground_truth_mu"] for r in rows],
            [r["predicted_mu"] for r in rows],
            c=[r["iteration"] for r in rows],
            s=8,
            alpha=0.25,
        )
        ax.plot([0.3, 1.2], [0.3, 1.2], "k--")
        ax.set(
            xlabel="Ground-truth foot friction μ",
            ylabel="Estimated μ",
            title="Predicted vs. Ground-Truth Friction\nMeasured imagined samples; not an independent test set",
        )
        fig.colorbar(pts, ax=ax, label="Iteration")
        figure(fig, output / "04_predicted_friction")
        made.append("04_predicted_friction")
    else:
        missing.append("04_predicted_friction")
    if args.eval_dir:
        ed = output_path(args.eval_dir)
        rows = json.loads((ed / "episodes.json").read_text())
        summary = []
        for variant in ["base", "adapted", "residual_disabled"]:
            for mu in sorted({r["mu"] for r in rows}):
                rs = [r for r in rows if r["variant"] == variant and r["mu"] == mu]
                if not rs:
                    continue
                summary.append(
                    {
                        "variant": variant,
                        "mu": mu,
                        "episodes": len(rs),
                        "failed": sum(r["failed"] for r in rs),
                        "mean_norm_return": float(np.mean([r["norm_return"] for r in rs])),
                        "failure_rate": float(np.mean([r["failed"] for r in rs])),
                        "friction_mae": None
                        if variant == "base"
                        else float(np.mean([r["friction_mae"] for r in rs])),
                        "capacity_affected_episodes": sum(
                            r["capacity_overflow_steps"] > 0 for r in rs
                        ),
                    }
                )
        save_json(output / "evaluation_summary.json", summary)
        for key, title, ylabel, name in [
            (
                "mean_norm_return",
                "Tracking Reward vs. Friction Coefficient",
                "Mean normalized full-clip return",
                "05_tracking_vs_friction",
            ),
            (
                "failure_rate",
                "Failure Rate vs. Friction Coefficient",
                "Failure rate",
                "06_failure_vs_friction",
            ),
        ]:
            fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
            for variant in ["base", "adapted", "residual_disabled"]:
                rs = [r for r in summary if r["variant"] == variant]
                ax.plot(
                    [r["mu"] for r in rs],
                    [r[key] for r in rs],
                    "o-",
                    label=f"{variant} ({rs[0]['episodes']} clips/μ)" if rs else variant,
                )
            ax.set(
                xlabel="Effective foot–ground friction coefficient μ",
                ylabel=ylabel,
                title=title
                + "\nStandard LAFAN1 benchmark; inspect capacity flags before interpretation",
            )
            ax.legend()
            ax.grid(alpha=0.2)
            if min(r["mu"] for r in summary) < 0.3:
                ax.axvspan(min(r["mu"] for r in summary) - 0.01, 0.3, color="orange", alpha=0.12)
                ax.text(
                    0.02,
                    0.97,
                    "μ < 0.3: OOD stress test",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=9,
                )
            figure(fig, output / name)
            made.append(name)
    else:
        missing += ["05_tracking_vs_friction", "06_failure_vs_friction"]
    manifest = {
        "generated": made,
        "missing_not_fabricated": missing,
        "validation_status": None if check is None else check["status"],
        "fine_tuning_results_present": metrics.exists(),
        "evaluation_present": args.eval_dir is not None,
    }
    save_json(output / "manifest.json", manifest)
    text = [
        "# Experiment 009 status report",
        "",
        f"Generated plots: {made}.",
        f"Not generated (no measurements): {missing}.",
        "",
        (
            "Actual training and paired LAFAN1 evaluation measurements are available. See evaluation_summary.json and RESULTS_ZH.md for measured differences, numerical flags and interpretation limits."
            if args.eval_dir
            else "Tracking improvement cannot be assessed without paired physical evaluation. Validation-only scratch updates do not count as fine-tuning results."
        ),
    ]
    if check:
        text += [
            "",
            f"Full-size validation status: {check['status']}.",
            f"Full update measurements: {check.get('full_update', {})}.",
            f"Frozen hashes: {check.get('frozen_hashes', {})}.",
        ]
    (output / "REPORT.md").write_text("\n".join(text) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
