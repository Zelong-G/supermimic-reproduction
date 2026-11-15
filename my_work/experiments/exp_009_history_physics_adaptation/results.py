"""Measured paired evaluation, Encoder diagnostics and Chinese research findings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT, ROOT, output_path, save_json


def table(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def figure(fig, path):
    for ext in ["png", "pdf"]:
        fig.savefig(path.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def paired(rows, mu, clean=False):
    lookup = {(r["variant"], r["name"]): r for r in rows if r["mu"] == mu}
    names = sorted(r["name"] for r in rows if r["mu"] == mu and r["variant"] == "base")
    result = []
    for name in names:
        b, a, d = [lookup[v, name] for v in ["base", "adapted", "residual_disabled"]]
        assert b["initial_state_sha256"] == a["initial_state_sha256"] == d["initial_state_sha256"]
        if clean and any(r["capacity_overflow_steps"] for r in [b, a, d]):
            continue
        result.append(
            {
                "name": name,
                "mu": mu,
                "base_return": b["norm_return"],
                "adapted_return": a["norm_return"],
                "disabled_return": d["norm_return"],
                "delta_adapted_vs_base": a["norm_return"] - b["norm_return"],
                "delta_disabled_vs_base": d["norm_return"] - b["norm_return"],
                "capacity_affected": any(r["capacity_overflow_steps"] > 0 for r in [b, a, d]),
            }
        )
    return result


def paired_summary(pairs):
    if not pairs:
        return {
            "episodes": 0,
            "mean_delta": None,
            "bootstrap_low": None,
            "bootstrap_high": None,
            "mean_abs_disabled_delta": None,
        }
    x = np.array([r["delta_adapted_vs_base"] for r in pairs])
    rng = np.random.default_rng(9)
    boot = x[rng.integers(0, len(x), (2000, len(x)))].mean(-1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "episodes": len(x),
        "mean_delta": float(x.mean()),
        "bootstrap_low": float(lo),
        "bootstrap_high": float(hi),
        "mean_abs_disabled_delta": float(
            np.mean([abs(r["delta_disabled_vs_base"]) for r in pairs])
        ),
    }


def analyze(run, evaluation):
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    protocol = json.loads((evaluation / "protocol.json").read_text())
    rows = json.loads((evaluation / "episodes.json").read_text())
    predictions = json.loads((evaluation / "friction_predictions.json").read_text())
    training = json.loads((run / "training_summary.json").read_text())
    frictions = sorted(protocol["frictions"], reverse=True)
    n = len(protocol["clips"])
    assert len(rows) == 3 * n * len(frictions), "Evaluation is incomplete"
    summaries = []
    effects = []
    pairs = []
    encoder = []
    for mu in frictions:
        for variant in ["base", "adapted", "residual_disabled"]:
            r = [r for r in rows if r["mu"] == mu and r["variant"] == variant]
            assert len(r) == n
            summaries.append(
                {
                    "mu": mu,
                    "variant": variant,
                    "episodes": n,
                    "failures": sum(r["failed"] for r in r),
                    "mean_return": float(np.mean([r["norm_return"] for r in r])),
                    "mean_body_position_error_m": float(
                        np.mean([r["mean_body_position_error_m"] for r in r])
                    ),
                    "mean_joint_rmse_rad": float(np.mean([r["mean_joint_rmse_rad"] for r in r])),
                    "friction_mae": None
                    if variant == "base"
                    else float(np.mean([r["friction_mae"] for r in r])),
                    "capacity_affected_episodes": sum(r["capacity_overflow_steps"] > 0 for r in r),
                    "solver_limit_steps": sum(r["solver_limit_steps"] for r in r),
                }
            )
        p = paired(rows, mu)
        pairs += p
        effects.append({"mu": mu, "subset": "all_standard_episodes", **paired_summary(p)})
        effects.append(
            {
                "mu": mu,
                "subset": "common_capacity_clean",
                **paired_summary(paired(rows, mu, clean=True)),
            }
        )
        samples = [p for p in predictions if p["variant"] == "adapted" and p["mu"] == mu]
        values = np.array([p["predicted_mu"] for p in samples])
        names = sorted({p["name"] for p in samples})
        means = [
            np.mean([p["predicted_mu"] for p in samples if p["name"] == name]) for name in names
        ]
        mae = next(
            r["friction_mae"] for r in summaries if r["variant"] == "adapted" and r["mu"] == mu
        )
        encoder.append(
            {
                "true_mu": mu,
                "episodes": n,
                "sampled_frames": len(samples),
                "mean_estimate_equal_clip": float(np.mean(means)),
                "mean_estimate_sample_weighted": float(values.mean()),
                "std_sample_weighted": float(values.std()),
                "minimum": float(values.min()),
                "p10": float(np.quantile(values, 0.1)),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)),
                "maximum": float(values.max()),
                "full_step_mae_equal_episode": mae,
                "training_prior_constant_0_75_mae": abs(mu - 0.75),
            }
        )
    table(out / "comparison.csv", summaries)
    table(out / "paired_score_deltas.csv", pairs)
    table(out / "paired_effects.csv", effects)
    table(out / "encoder_outputs.csv", encoder)
    result = {
        "iteration": protocol["iteration"],
        "clips_per_condition": n,
        "episodes": len(rows),
        "protocol": protocol,
        "comparison": summaries,
        "paired_effects": effects,
        "encoder": encoder,
        "training": training,
        "interpretation_limits": [
            "Single deterministic seed and selected fixed benchmark; bootstrap resamples motion episodes, not frames.",
            "Capacity-clean subset is conditional on all three conditions being clean and can exclude difficult cases.",
            "Solver-limit flags are retained; residual-disabled differences quantify one numerical repeatability reference, not a population noise estimate.",
            "Encoder distributions use half-second sampled real observations; MAE uses every executed real observation with equal episode weights.",
            "Training-prior mean 0.75 is a diagnostic constant predictor, not an inference ablation of the trained controller.",
            "Tracking differences cannot isolate friction conditioning from extra residual capacity without additional controls.",
        ],
    }
    save_json(out / "results.json", result)
    # Encoder values on real held-out trajectories, separate from imagined training samples.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    ordered = sorted(encoder, key=lambda r: r["true_mu"])
    axes[0].plot(
        [r["true_mu"] for r in ordered],
        [r["mean_estimate_equal_clip"] for r in ordered],
        "o-",
        label="Trained Encoder (equal clip mean)",
    )
    axes[0].plot([0.3, 1], [0.3, 1], "k--", label="Identity")
    axes[0].axhline(0.75, color="gray", ls=":", label="Training-prior constant")
    axes[0].set(xlabel="True foot friction μ", ylabel="Estimated foot friction μ", ylim=(0.3, 1.2))
    axes[0].legend(fontsize=8)
    arrays = [
        np.array(
            [p["predicted_mu"] for p in predictions if p["variant"] == "adapted" and p["mu"] == mu]
        )
        for mu in sorted(frictions)
    ]
    axes[1].boxplot(
        arrays, tick_labels=[str(mu) for mu in sorted(frictions)], whis=(10, 90), showfliers=False
    )
    axes[1].set(
        xlabel="True foot friction μ",
        ylabel="Real-observation estimated μ",
        ylim=(0.3, 1.2),
        title="Median / IQR; whiskers: 10th–90th percentile",
    )
    fig.suptitle(
        f"Physics Encoder on held-out LAFAN1: {n} clips per μ\nSampled every 0.5 s; distribution is duration-weighted; frozen checkpoint {protocol['iteration']}"
    )
    figure(fig, out / "08_encoder_real_outputs")
    selected = (
        "walk1_subject1"
        if any(p["name"] == "walk1_subject1" for p in predictions)
        else predictions[0]["name"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, mu in zip(axes.flat, sorted(frictions)):
        ps = sorted(
            [
                p
                for p in predictions
                if p["variant"] == "adapted" and p["name"] == selected and p["mu"] == mu
            ],
            key=lambda p: p["step"],
        )
        ax.plot(
            [p["step"] * 0.02 for p in ps], [p["predicted_mu"] for p in ps], label="Estimated μ"
        )
        ax.axhline(mu, color="black", ls="--", label="True μ")
        ax.set(
            xlabel="Episode time (s)",
            ylabel="Estimated friction μ",
            ylim=(0.3, 1.2),
            title=f"True μ = {mu:g}",
        )
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        f"Real Physics Encoder output over time: {selected}\nPrediction changes alone do not establish friction identification"
    )
    figure(fig, out / "09_encoder_real_timeseries")
    lines = [
        "# Experiment 009：5,000 次联合微调与 LAFAN1 评测",
        "",
        "本报告只使用实际测量。BONES-SEED 用于训练；LAFAN1 未参与训练或本次 checkpoint 选择。使用固定最终 5,000 次检查点，没有启动第二次微调。",
        "",
        "## 训练与权重变化",
        "",
        f"训练恢复后的循环时间：{training['training_loop_seconds_last_session'] / 60:.2f} 分钟（不含初始化/replay 填充及之前的验证）。实际迭代耗时合计：{training['sum_iteration_seconds'] / 60:.2f} 分钟；保存/日志等开销另计。",
        "",
        f"原 Policy、Motion Encoder、WM 与归一化状态冻结，初末记录的哈希一致；原预训练 checkpoint 文件 SHA-256 未变。新增 Encoder 的权重变化 L2={training['new_module_weight_changes']['encoder']['weight_change_l2']:.4f}，Adapter 为 {training['new_module_weight_changes']['adapter']['weight_change_l2']:.4f}。Critic 按原算法更新；这些权重范数不代表质量提升。",
        "",
        f"训练最初/最后 200 次平均摩擦 MAE：{training['first_200_mean']['friction_mae']:.4f} → {training['last_200_mean']['friction_mae']:.4f}；原想象目标 J：{training['first_200_mean']['imag_objective']:.4f} → {training['last_200_mean']['imag_objective']:.4f}。这些是想象轨迹训练指标，不能替代真实仿真。",
        "",
        "## 真实仿真比较",
        "",
        f"每个 μ、每个策略均为 {n} 条完整动作 /{n} 个 Episode，共 {len(rows)} 个 Episode。基准协议、初始状态与 seed 配对；仅脚部切向摩擦改变。",
        "",
        "| μ | Base 回报 | Adapted 回报 | Disabled 回报 | Base/Adapted/Disabled 失败数 | Δ回报及描述性95%区间 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for mu in frictions:
        b, a, d = [
            next(r for r in summaries if r["mu"] == mu and r["variant"] == v)
            for v in ["base", "adapted", "residual_disabled"]
        ]
        e = next(r for r in effects if r["mu"] == mu and r["subset"] == "all_standard_episodes")
        lines.append(
            f"| {mu:g} | {b['mean_return']:.4f} | {a['mean_return']:.4f} | {d['mean_return']:.4f} | {b['failures']}/{a['failures']}/{d['failures']}（各n={n}） | {e['mean_delta']:+.4f} [{e['bootstrap_low']:+.4f}, {e['bootstrap_high']:+.4f}] |"
        )
    lines += [
        "",
        "## 数值有效性",
        "",
        "| μ | Base/Adapted/Disabled 容量标记 Episode 数 | 三条件共同无容量错误的样本数 | 干净子集平均 Δ回报 | Disabled–Base 平均绝对差 |",
        "|---|---|---:|---:|---:|",
    ]
    for mu in frictions:
        counts = [
            next(
                r["capacity_affected_episodes"]
                for r in summaries
                if r["mu"] == mu and r["variant"] == v
            )
            for v in ["base", "adapted", "residual_disabled"]
        ]
        e = next(r for r in effects if r["mu"] == mu and r["subset"] == "common_capacity_clean")
        fmt = lambda x: "N/A" if x is None else f"{x:.5f}"
        lines.append(
            f"| {mu:g} | {'/'.join(map(str, counts))} | {e['episodes']} | {fmt(e['mean_delta'])} | {fmt(e['mean_abs_disabled_delta'])} |"
        )
    lines += [
        "",
        "主表保留标准 benchmark 全部 Episode，不把带容量错误的失败直接归因于 Policy。共同干净子集是条件性诊断，可能排除难动作；不能替代完整 benchmark。求解器上限标记另在 comparison.csv 中报告。Disabled 在相同观测下精确恢复原动作，但重复 GPU 接触求解可能产生轨迹差异。",
        "",
        "## Physics Encoder 实际输出",
        "",
        "| 真实 μ | 预测均值（动作等权） | 预测中位数 | 预测10%–90% | 全步 MAE（Episode等权） | 常数0.75的MAE |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for r in encoder:
        lines.append(
            f"| {r['true_mu']:g} | {r['mean_estimate_equal_clip']:.4f} | {r['median']:.4f} | {r['p10']:.4f}–{r['p90']:.4f} | {r['full_step_mae_equal_episode']:.4f} | {r['training_prior_constant_0_75_mae']:.4f} |"
        )
    prior = float(np.mean([r["training_prior_constant_0_75_mae"] for r in encoder]))
    learned = float(np.mean([r["full_step_mae_equal_episode"] for r in encoder]))
    lines += [
        "",
        f"四个摩擦条件等权平均 MAE：Encoder={learned:.4f}，训练先验常数预测器={prior:.4f}。常数基线只是估计准确性的参照，不是另一个训练实验，也不是 Adapter 的条件输入消融。",
        "",
        "## 结论边界",
        "",
    ]
    for mu in [1.0, 0.3]:
        e = next(r for r in effects if r["mu"] == mu and r["subset"] == "all_standard_episodes")
        lines.append(
            f"- μ={mu:g}：Adapted 相对 Base 的平均归一化回报变化为 {e['mean_delta']:+.4f}；该数值及容量/数值重复性检查应一起解释。"
        )
    lines += [
        "- 训练 MAE 下降不证明未见动作上的摩擦估计改善；应以本表真实观测 MAE 和预测分布判断。",
        "- 跟踪改善也不能单独证明 Encoder 推断了摩擦或该标量改善了控制；仍需常数/打乱估计等额外推断对照或独立消融。",
        "- 当前结果不能推断“继续六小时一定更好”或“已经过拟合”。若继续，应保留本次5k基线并预先固定比较检查点，避免事后只挑最好结果。",
        "- Body/joint 误差是在实际存活步上统计；早停时长不同，直接比较存在选择效应。脚部滑移速度未可靠实现，不编造滑移指标。",
        "",
        "## 英文图表",
        "",
        "![Tracking](05_tracking_vs_friction.png)",
        "![Failure](06_failure_vs_friction.png)",
        "![Real Encoder](08_encoder_real_outputs.png)",
        "![Encoder time series](09_encoder_real_timeseries.png)",
        "",
        "所有图均有 PDF。训练损失、奖励、训练 MAE、训练预测与残差幅度见 01–04、07 图。完整原始 Episode CSV/JSON 和预测帧数据位于对应 evaluation 目录；本目录保留 comparison.csv、paired_effects.csv、encoder_outputs.csv 和 results.json。",
    ]
    (out / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n")
    save_json(
        OUT / "LATEST_RUN.json",
        {
            "state": "evaluated",
            "run_dir": str(run.relative_to(ROOT)),
            "evaluation_dir": str(evaluation.relative_to(ROOT)),
            "report": str((out / "RESULTS_ZH.md").relative_to(ROOT)),
            "iteration": protocol["iteration"],
        },
    )
    print(json.dumps({"comparison": summaries, "encoder": encoder}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    a = p.parse_args()
    analyze(output_path(a.run_dir), output_path(a.eval_dir))


if __name__ == "__main__":
    main()
