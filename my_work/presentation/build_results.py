"""Build a small, source-linked PPT handoff from saved results; no GPU/training."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "my_work/outputs"
E9 = OUT / "exp_009_history_physics_adaptation"
SOURCES = set()


def read_json(path):
    SOURCES.add(path)
    return json.loads(path.read_text())


def read_csv(path):
    SOURCES.add(path)
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def savefig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(HERE / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    fixed, effects, encoders = [], [], []
    for run in ["sweep_5k_10k_20k_80862", "final_vs_community_new_frictions"]:
        folder = E9 / "evaluation" / run / "analysis"
        for row in read_csv(folder / "comparison.csv"):
            if row["variant"] in ["base", "step_80862"]:
                fixed.append(
                    {
                        "mu": float(row["mu"]),
                        "variant": row["variant"],
                        "episodes": int(row["episodes"]),
                        "mean_return": float(row["mean_return"]),
                        "failures": int(row["failures"]),
                        "failure_rate": float(row["failure_rate"]),
                        "capacity_affected_episodes": int(row["capacity_affected_episodes"]),
                        "out_of_training_range": not (0.3 <= float(row["mu"]) <= 1.2),
                        "source_run": run,
                    }
                )
        for row in read_csv(folder / "paired_effects.csv"):
            if row["variant"] == "step_80862" and row["subset"] == "standard_all":
                effects.append(
                    {
                        "mu": float(row["mu"]),
                        "delta": float(row["delta"]),
                        "ci_low": float(row["low"]),
                        "ci_high": float(row["high"]),
                        "motions": int(row["n"]),
                        "source_run": run,
                    }
                )
        encoders += [
            {**row, "source_run": run}
            for row in read_csv(folder / "encoder_outputs.csv")
            if row["variant"] == "step_80862"
        ]
    fixed.sort(key=lambda r: (r["mu"], r["variant"]))
    effects.sort(key=lambda r: r["mu"])
    encoders.sort(key=lambda r: float(r["mu"]))
    assert len(fixed) == 20 and len({(r["mu"], r["variant"]) for r in fixed}) == 20
    write_csv(HERE / "fixed_friction_comparison.csv", fixed)
    write_csv(HERE / "fixed_friction_paired_effects.csv", effects)
    write_csv(HERE / "encoder_outputs.csv", encoders)
    for metric, ylabel, name in [
        ("mean_return", "Mean normalized tracking return", "01_fixed_tracking"),
        ("failure_rate", "Failure rate", "02_fixed_failure"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
        ax.axvspan(0.3, 1.2, color="gray", alpha=0.1, label="Training friction range")
        for v, label in [("base", "Community pretrained"), ("step_80862", "Final (80,862)")]:
            rows = [r for r in fixed if r["variant"] == v]
            ax.plot([r["mu"] for r in rows], [r[metric] for r in rows], "o-", label=label)
        ticks = sorted({r["mu"] for r in fixed})
        ax.set(
            xscale="log",
            xlabel="Effective foot–ground friction coefficient μ",
            ylabel=ylabel,
            title="Fixed-friction evaluation: 40 full LAFAN1 motions per condition\nTwo evaluation batches; paired initial states; single run per condition",
        )
        ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
        ax.minorticks_off()
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        savefig(fig, name)
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    xs = np.arange(len(effects))
    ys = [r["delta"] for r in effects]
    ax.errorbar(
        xs,
        ys,
        yerr=[
            [r["delta"] - r["ci_low"] for r in effects],
            [r["ci_high"] - r["delta"] for r in effects],
        ],
        fmt="o",
        capsize=4,
    )
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xticks(xs, [f"{r['mu']:g}" for r in effects])
    ax.set(
        xlabel="Friction coefficient μ",
        ylabel="Final − community tracking return",
        title="Paired effects: descriptive 95% motion-bootstrap intervals\n40 motions; intervals do not measure simulator repeat variability",
    )
    ax.grid(alpha=0.2)
    savefig(fig, "03_fixed_paired_effects")
    train = read_csv(E9 / "joint_6h_from5000/metrics.csv")
    metrics = ["friction_mae", "imag_reward_mean", "residual_abs_mean", "seconds_per_iteration"]
    windows = []
    for start in range(0, len(train), 1000):
        part = train[start : start + 1000]
        row = {
            "first_iteration": int(part[0]["iteration"]),
            "last_iteration": int(part[-1]["iteration"]),
            "updates": len(part),
        }
        for key in metrics:
            row[key + "_mean"] = float(np.mean([float(r[key]) for r in part]))
        windows.append(row)
    write_csv(HERE / "exp009_training_windows.csv", windows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key, label in zip(
        axes, metrics[:2], ["Imagined-rollout friction MAE", "Imagined per-step reward"]
    ):
        ax.plot([r["last_iteration"] for r in windows], [r[key + "_mean"] for r in windows])
        ax.set(xlabel="Global update", ylabel=label)
        ax.grid(alpha=0.2)
    fig.suptitle("Six-hour continuation: training diagnostics, not held-out performance")
    savefig(fig, "04_exp009_training")
    random = read_json(E9 / "evaluation/random_friction_final_vs_community/analysis/results.json")
    summary = {
        "fixed_friction": fixed,
        "fixed_paired_effects": effects,
        "encoder_outputs": encoders,
        "random_friction": {
            k: random[k]
            for k in [
                "complete",
                "summary",
                "effects",
                "usable_switch_windows",
                "total_switch_windows",
            ]
        },
        "random_prefix_audit": {
            "episodes": len(random["audits"]),
            "passed": sum(r["passed"] for r in random["audits"]),
        },
        "exp007_training": read_json(
            OUT / "exp_007_full_training/full_seed0/training_summary.json"
        ),
        "exp007_lafan1": read_csv(
            OUT / "exp_007_full_training/full_seed0/evaluation/lafan1_comparison.csv"
        ),
        "exp007_bones": read_csv(
            OUT / "exp_007_full_training/full_seed0/evaluation/bones_comparison.csv"
        ),
        "exp006": read_csv(OUT / "exp_006_bones_benchmark/summary.csv"),
        "exp008_fixed": read_csv(OUT / "exp_008_friction_adaptation/fixed_summary.csv"),
        "exp009_training": {
            "first_iteration": int(train[0]["iteration"]),
            "final_iteration": int(train[-1]["iteration"]),
            "updates": len(train),
            "loop_elapsed_seconds": float(train[-1]["elapsed_seconds"]),
            "last_1000_mean": {
                k: float(np.mean([float(r[k]) for r in train[-1000:]])) for k in metrics
            },
        },
        "interpretation": "No established overall friction robustness or online adaptation advantage. Hypotheses must be distinguished from observations.",
    }
    (HERE / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    sources = [
        {
            "path": str(p.relative_to(ROOT)),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(SOURCES)
    ]
    (HERE / "source_manifest.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
