"""Measured checkpoint comparison; partial sweeps are labeled and never filled in."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import output_path, save_json

LABELS = {
    "base": "Original",
    "residual_disabled": "Residual disabled",
    "step_5000": "5k",
    "step_10000": "10k",
    "step_20000": "20k",
    "step_80862": "80,862",
}
STEPS = ["step_5000", "step_10000", "step_20000", "step_80862"]


def csv_file(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def paired_effect(base, adapted, disabled=None, clean=False):
    b = {r["name"]: r for r in base}
    a = {r["name"]: r for r in adapted}
    d = {r["name"]: r for r in disabled or []}
    if set(b) != set(a):
        raise ValueError("Unmatched motions")
    delta = []
    noise = []
    for name in sorted(b):
        cohort = [b[name], a[name]] + ([d[name]] if disabled else [])
        if len({r["initial_state_sha256"] for r in cohort}) != 1:
            raise ValueError("Unpaired initial states")
        if clean and any(r["capacity_overflow_steps"] for r in cohort):
            continue
        delta.append(a[name]["norm_return"] - b[name]["norm_return"])
        if disabled:
            noise.append(abs(d[name]["norm_return"] - b[name]["norm_return"]))
    if not delta:
        return {"n": 0, "delta": None, "low": None, "high": None, "disabled_absolute_delta": None}
    x = np.array(delta)
    rng = np.random.default_rng(9)
    boot = x[rng.integers(0, len(x), (2000, len(x)))].mean(-1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "n": len(x),
        "delta": float(x.mean()),
        "low": float(low),
        "high": float(high),
        "disabled_absolute_delta": float(np.mean(noise)) if noise else None,
    }


def save_figure(fig, path):
    for ext in ["png", "pdf"]:
        fig.savefig(path.with_suffix("." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


def report(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    out = directory / "analysis"
    out.mkdir(exist_ok=True)
    payloads = [json.loads(p.read_text()) for p in sorted((directory / "groups").glob("*.json"))]
    lookup = {(p["variant"], p["mu"]): p for p in payloads}
    summary = []
    effects = []
    encoder = []
    expected = len(manifest["variants"]) * len(manifest["frictions"])
    complete = len(payloads) == expected
    for p in payloads:
        rows = p["episodes"]
        v = p["variant"]
        mu = p["mu"]
        summary.append(
            {
                "variant": v,
                "mu": mu,
                "episodes": len(rows),
                "mean_return": float(np.mean([r["norm_return"] for r in rows])),
                "failures": sum(r["failed"] for r in rows),
                "failure_rate": float(np.mean([r["failed"] for r in rows])),
                "mean_body_position_error_m": float(
                    np.mean([r["mean_body_position_error_m"] for r in rows])
                ),
                "mean_joint_rmse_rad": float(np.mean([r["mean_joint_rmse_rad"] for r in rows])),
                "friction_mae": None
                if v == "base"
                else float(np.mean([r["friction_mae"] for r in rows])),
                "capacity_affected_episodes": sum(r["capacity_overflow_steps"] > 0 for r in rows),
                "solver_limit_steps": sum(r["solver_limit_steps"] for r in rows),
                "wall_seconds": p["wall_seconds"],
                "other_gpu_process_seen_at_end": bool(p.get("other_gpu_processes_at_end")),
            }
        )
        if v in STEPS:
            samples = p["predictions"]
            values = np.array([s["predicted_mu"] for s in samples])
            names = sorted({s["name"] for s in samples})
            means = [
                np.mean([s["predicted_mu"] for s in samples if s["name"] == name]) for name in names
            ]
            encoder.append(
                {
                    "variant": v,
                    "mu": mu,
                    "sampled_frames": len(samples),
                    "mean_estimate_equal_motion": float(np.mean(means)),
                    "median_sample_weighted": float(np.median(values)),
                    "p10": float(np.quantile(values, 0.1)),
                    "p90": float(np.quantile(values, 0.9)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "full_step_mae_equal_episode": float(
                        np.mean([r["friction_mae"] for r in rows])
                    ),
                    "constant_0_75_mae": abs(mu - 0.75),
                }
            )
            if ("base", mu) in lookup:
                disabled = lookup.get(("residual_disabled", mu), {}).get("episodes")
                for clean in [False, True]:
                    effects.append(
                        {
                            "variant": v,
                            "mu": mu,
                            "subset": "common_capacity_clean" if clean else "standard_all",
                            **paired_effect(lookup["base", mu]["episodes"], rows, disabled, clean),
                        }
                    )
    csv_file(out / "comparison.csv", summary)
    csv_file(out / "paired_effects.csv", effects)
    csv_file(out / "encoder_outputs.csv", encoder)
    result = {
        "complete": complete,
        "completed_groups": len(payloads),
        "expected_groups": expected,
        "completed_episodes": sum(r["episodes"] for r in summary),
        "summary": summary,
        "effects": effects,
        "encoder": encoder,
        "manifest": manifest,
    }
    save_json(out / "results.json", result)
    for metric, title, ylabel, name in [
        (
            "mean_return",
            "Tracking Reward vs. Friction",
            "Mean normalized full-motion return",
            "01_tracking_vs_friction",
        ),
        ("failure_rate", "Failure Rate vs. Friction", "Failure rate", "02_failure_vs_friction"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        for variant in ["base"] + STEPS + ["residual_disabled"]:
            rs = sorted([r for r in summary if r["variant"] == variant], key=lambda r: r["mu"])
            if rs:
                ax.plot(
                    [r["mu"] for r in rs],
                    [r[metric] for r in rs],
                    "o--" if variant == "residual_disabled" else "o-",
                    label=f"{LABELS[variant]} (n={rs[0]['episodes']}/μ)",
                )
        ax.set(
            xlabel="Effective foot–ground friction coefficient μ",
            ylabel=ylabel,
            title=title
            + ("\nComplete sweep" if complete else "\nPARTIAL: only completed conditions shown"),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        save_figure(fig, out / name)
    if encoder:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for variant in STEPS:
            rs = sorted([r for r in encoder if r["variant"] == variant], key=lambda r: r["mu"])
            if not rs:
                continue
            axes[0].plot(
                [r["mu"] for r in rs],
                [r["mean_estimate_equal_motion"] for r in rs],
                "o-",
                label=LABELS[variant],
            )
            axes[1].plot(
                [r["mu"] for r in rs],
                [r["full_step_mae_equal_episode"] for r in rs],
                "o-",
                label=LABELS[variant],
            )
        bounds = [min(manifest["frictions"]), max(manifest["frictions"])]
        axes[0].plot(bounds, bounds, "k--", label="Identity")
        axes[0].set(xlabel="True μ", ylabel="Estimated μ (equal motion mean)", ylim=(0.3, 1.2))
        xs = sorted(manifest["frictions"])
        axes[1].plot(xs, [abs(x - 0.75) for x in xs], "k--", label="Constant training-prior mean")
        axes[1].set(xlabel="True μ", ylabel="Friction MAE (all real steps, equal episodes)")
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.suptitle(
            "Physics Encoder on real held-out LAFAN1 observations"
            + ("" if complete else " — PARTIAL")
        )
        save_figure(fig, out / "03_encoder_outputs")
    lines = [
        "# Experiment 009：延长训练检查点评测",
        "",
        f"状态：{'完整' if complete else '部分完成，不能作为最终结论'}；已完成 {len(payloads)}/{expected} 个条件，{result['completed_episodes']} 个 Episode。",
        "",
        "比较策略：" + "、".join(LABELS[v] for v in manifest["variants"]) + "。没有新训练；使用相同 LAFAN1 完整动作、seed=0、原始物理设置和配对初始状态。仅在包含 disabled 对照时提供数值重复性参照。",
        "训练摩擦支持范围为 [0.3, 1.2]；范围内新取值不等于范围外泛化。Encoder 输出被限制在 [0.3, 1.2]，无法准确预测范围外真值；范围外控制鲁棒性与参数估计准确性应分别解释。",
        "",
        "## 跟踪表现",
        "",
        "| μ | 策略/检查点 | 平均归一化回报 | 失败数 / Episode数 | 容量标记 Episode数 |",
        "|---|---|---:|---:|---:|",
    ]
    for mu in manifest["frictions"]:
        for variant in ["base"] + STEPS + ["residual_disabled"]:
            rs = [r for r in summary if r["variant"] == variant and r["mu"] == mu]
            if rs:
                r = rs[0]
                lines.append(
                    f"| {mu:g} | {LABELS[variant]} | {r['mean_return']:.4f} | {r['failures']}/{r['episodes']} | {r['capacity_affected_episodes']} |"
                )
    lines += [
        "",
        "## 相对原策略的配对回报变化",
        "",
        "区间为按动作 Episode 重采样的描述性 bootstrap 95% 区间；没有把连续帧当作独立样本。",
        "",
        "| μ | 检查点 | Δ回报 | 区间 | Disabled–Base平均绝对差 |",
        "|---|---|---:|---|---:|",
    ]
    for r in effects:
        if r["subset"] != "standard_all":
            continue
        noise = (
            ("未设置" if "residual_disabled" not in manifest["variants"] else "待完成")
            if r["disabled_absolute_delta"] is None
            else f"{r['disabled_absolute_delta']:.4f}"
        )
        lines.append(
            f"| {r['mu']:g} | {LABELS[r['variant']]} | {r['delta']:+.4f} | [{r['low']:+.4f}, {r['high']:+.4f}] | {noise} |"
        )
    lines += [
        "",
        "## Physics Encoder 输出",
        "",
        "均值按动作等权；分位数来自每0.5秒采样的真实观测，受动作时长和提前失败影响。MAE使用每一个实际执行步，再按 Episode 等权。",
        "",
        "| μ | 检查点 | 预测均值 | 预测10%–90% | 全步MAE | 常数0.75 MAE |",
        "|---|---|---:|---|---:|---:|",
    ]
    for r in sorted(encoder, key=lambda r: (-r["mu"], STEPS.index(r["variant"]))):
        lines.append(
            f"| {r['mu']:g} | {LABELS[r['variant']]} | {r['mean_estimate_equal_motion']:.4f} | {r['p10']:.4f}–{r['p90']:.4f} | {r['full_step_mae_equal_episode']:.4f} | {r['constant_0_75_mae']:.4f} |"
        )
    lines += [
        "",
        "## 结论限制",
        "",
        "- 没有以 LAFAN1 调整训练损失或选择新的训练参数；所有指定检查点都报告，不事后只挑最好值。",
        "- 数值容量标记与求解器迭代上限分开记录。带容量错误的失败不能直接归因于模型；共同干净子集的结果保留在 paired_effects.csv，属于条件性诊断。",
        "- Disabled 与原策略在相同观测下动作精确一致，但长序列重复仿真仍可能分歧。单个重复对照不足以完整估计数值变异；小变化需要更多重复或数值控制。",
        "- Encoder 输出变化或跟踪改善均不能单独证明识别了摩擦、或控制收益来自这个标量；还需要常数/打乱估计等推断对照。",
        "- 训练指标退化或验证差异不能单独证明过拟合；采样分布、模型偏差与数值变化都需区分。",
        "- 当前脚部摩擦干预与训练 ξ 定义一致，不是全身所有接触都采用统一低摩擦。滑移速度未可靠测量，不编造数值。",
        "",
        "## 图表",
        "",
        "![Tracking](01_tracking_vs_friction.png)",
        "![Failure](02_failure_vs_friction.png)",
        "![Encoder](03_encoder_outputs.png)",
        "",
        "图表同时提供 PDF；原始 group JSON 含逐动作结果、真实预测帧、接触证据、运行耗时与数值标记。",
    ]
    (out / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "complete": complete,
                "groups": len(payloads),
                "episodes": result["completed_episodes"],
                "report": str(out / "RESULTS_ZH.md"),
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, required=True)
    a = p.parse_args()
    report(output_path(a.eval_dir))


if __name__ == "__main__":
    main()
