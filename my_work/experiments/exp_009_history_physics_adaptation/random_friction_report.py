"""Report random-friction robustness and strictly paired within-policy responses."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import file_hash, output_path, save_json
from random_friction_protocol import DT, SWITCH_STEPS, clustered_interval, prefix_audit, schedule


def csv_save(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def figure(fig, path):
    for ext in ["png", "pdf"]:
        fig.savefig(path.with_suffix("." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


def report(directory):
    m = json.loads((directory / "manifest.json").read_text())
    out = directory / "analysis"
    out.mkdir(exist_ok=True)
    groups = {}
    for p in sorted((directory / "groups").glob("*.json")):
        payload = json.loads(p.read_text())
        if file_hash(p.with_suffix(".npz")) != payload["trace_sha256"]:
            raise AssertionError("Trace checksum mismatch")
        groups[p.stem] = payload
    episodes, audits, windows, summaries = [], [], [], []
    # Only 0..1s following a switch, so changes at 3s do not overlap windows.
    event_curves = defaultdict(list)
    example = None
    for variant in m["variants"]:
        control_key = f"{variant}_control"
        if control_key not in groups:
            continue
        crows = groups[control_key]["episodes"]
        control = np.load(directory / "groups" / f"{control_key}.npz")
        try:
            episodes.extend(
                {
                    **r,
                    "prefix_gate_passed": None,
                    "return_delta_vs_own_control": 0.0,
                    "control_capacity_overflow_steps": r["capacity_overflow_steps"],
                }
                for r in crows
            )
            for seed in m["sequence_seeds"]:
                key = f"{variant}_random_seed{seed}"
                if key not in groups:
                    continue
                data = np.load(directory / "groups" / f"{key}.npz")
                try:
                    for i, row in enumerate(groups[key]["episodes"]):
                        if row["name"] != crows[i]["name"]:
                            raise AssertionError("Motion ordering changed")
                        expected = schedule(row["name"], seed, len(data["schedule"]))
                        if not np.array_equal(data["schedule"][:, i], expected):
                            raise AssertionError(
                                "Saved random sequence differs from prescribed paired schedule"
                            )
                        if not np.array_equal(
                            data["mu"][: row["steps"], i], expected[: row["steps"]]
                        ):
                            raise AssertionError(
                                "Executed friction differs from prescribed schedule"
                            )
                        if not np.all(control["mu"][: crows[i]["steps"], i] == 1):
                            raise AssertionError("No-switch control friction changed")
                        paired = row["initial_state_sha256"] == crows[i]["initial_state_sha256"]
                        audit = prefix_audit(control["prefix"][:, i], data["prefix"][:, i])
                        audit.update(
                            name=row["name"],
                            variant=variant,
                            sequence_seed=seed,
                            initial_state_paired=paired,
                        )
                        audit["passed"] &= (
                            paired
                            and row["steps"] >= SWITCH_STEPS
                            and crows[i]["steps"] >= SWITCH_STEPS
                        )
                        audits.append(audit)
                        rr = dict(row)
                        rr["prefix_gate_passed"] = audit["passed"]
                        rr["return_delta_vs_own_control"] = (
                            row["norm_return"] - crows[i]["norm_return"]
                        )
                        rr["control_capacity_overflow_steps"] = crows[i]["capacity_overflow_steps"]
                        episodes.append(rr)
                        common = min(row["steps"], crows[i]["steps"])
                        clean = not (
                            row["capacity_overflow_steps"] or crows[i]["capacity_overflow_steps"]
                        )
                        for start in range(SWITCH_STEPS, row["steps"], SWITCH_STEPS):
                            end = start + 50
                            complete = end <= common
                            usable = complete and audit["passed"] and clean
                            delta_p = delta_o = None
                            if usable:
                                dp = (
                                    data["position_error"][start:end, i]
                                    - control["position_error"][start:end, i]
                                )
                                do = (
                                    data["orientation_error"][start:end, i]
                                    - control["orientation_error"][start:end, i]
                                )
                                delta_p, delta_o = float(dp.mean()), float(do.mean())
                                event_curves[variant, row["name"]].append(dp)
                            windows.append(
                                {
                                    "name": row["name"],
                                    "variant": variant,
                                    "sequence_seed": seed,
                                    "switch_time_s": start * DT,
                                    "from_mu": float(data["mu"][start - 1, i]),
                                    "to_mu": float(data["mu"][start, i]),
                                    "complete_common_1s_window": complete,
                                    "prefix_gate_passed": audit["passed"],
                                    "capacity_clean": clean,
                                    "usable_response_window": usable,
                                    "excess_position_m": delta_p,
                                    "excess_orientation_rad": delta_o,
                                }
                            )
                        if (
                            example is None
                            and variant == "step_80862"
                            and seed == m["sequence_seeds"][0]
                        ):
                            n = row["steps"]
                            example = {
                                k: data[k][:n, i].copy()
                                for k in [
                                    "time",
                                    "mu",
                                    "predicted_mu",
                                    "position_error",
                                    "orientation_error",
                                    "action_change",
                                ]
                            }
                            example["name"] = row["name"]
                finally:
                    data.close()
        finally:
            control.close()
    random = [r for r in episodes if not r["control"]]
    for variant in m["variants"]:
        rows = [r for r in random if r["variant"] == variant]
        if not rows:
            continue
        per_motion = defaultdict(list)
        for r in rows:
            per_motion[r["name"]].append(r["norm_return"])
        low, high = clustered_interval([np.mean(v) for v in per_motion.values()])
        summaries.append(
            {
                "variant": variant,
                "episodes": len(rows),
                "motions": len(per_motion),
                "mean_return": float(np.mean([np.mean(v) for v in per_motion.values()])),
                "return_ci_low": low,
                "return_ci_high": high,
                "failures": sum(r["failed"] for r in rows),
                "failure_rate": float(np.mean([r["failed"] for r in rows])),
                "censored": sum(r["censored"] for r in rows),
                "prefix_gate_passed": sum(r["prefix_gate_passed"] for r in rows),
                "capacity_affected": sum(r["capacity_overflow_steps"] > 0 for r in rows),
            }
        )
    lookup = {(r["variant"], r["name"], r["sequence_seed"]): r for r in random}
    paired = []
    for r in random:
        if r["variant"] != "step_80862":
            continue
        b = lookup.get(("base", r["name"], r["sequence_seed"]))
        if b is None:
            continue
        if r["initial_state_sha256"] != b["initial_state_sha256"]:
            raise AssertionError("Cross-policy initial states differ")
        paired.append(
            {
                "name": r["name"],
                "sequence_seed": r["sequence_seed"],
                "return_delta": r["norm_return"] - b["norm_return"],
                "control_adjusted_delta": r["return_delta_vs_own_control"]
                - b["return_delta_vs_own_control"],
                "both_prefix_gates_passed": r["prefix_gate_passed"] and b["prefix_gate_passed"],
                "capacity_clean": not (
                    r["capacity_overflow_steps"]
                    or b["capacity_overflow_steps"]
                    or r["control_capacity_overflow_steps"]
                    or b["control_capacity_overflow_steps"]
                ),
            }
        )
    effects = {}
    for metric in ["return_delta", "control_adjusted_delta"]:
        for subset in ["all", "prefix_and_capacity_clean"]:
            by_motion = defaultdict(list)
            for r in paired:
                if subset != "all" and not (r["both_prefix_gates_passed"] and r["capacity_clean"]):
                    continue
                by_motion[r["name"]].append(r[metric])
            vals = [float(np.mean(v)) for v in by_motion.values()]
            effects[metric + "_" + subset] = {
                "motions": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "ci95": clustered_interval(vals),
            }
    csv_save(out / "episodes.csv", episodes)
    csv_save(out / "summary.csv", summaries)
    csv_save(out / "pairing_audit.csv", audits)
    csv_save(out / "switch_windows.csv", windows)
    csv_save(out / "paired_effects.csv", paired)
    expected = 2 * (1 + len(m["sequence_seeds"]))
    save_json(
        out / "results.json",
        {
            "manifest": m,
            "complete": len(groups) == expected,
            "summary": summaries,
            "effects": effects,
            "audits": audits,
            "usable_switch_windows": sum(r["usable_response_window"] for r in windows),
            "total_switch_windows": len(windows),
        },
    )
    if summaries:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        names = ["Original" if r["variant"] == "base" else "Final (80,862)" for r in summaries]
        means = [r["mean_return"] for r in summaries]
        axes[0].bar(names, means)
        axes[0].errorbar(
            range(len(means)),
            means,
            yerr=[
                [v - r["return_ci_low"] for v, r in zip(means, summaries)],
                [r["return_ci_high"] - v for v, r in zip(means, summaries)],
            ],
            fmt="none",
            color="k",
            capsize=4,
        )
        axes[0].set(
            ylabel="Mean normalized full-motion return", title="Tracking under random friction"
        )
        axes[1].bar(names, [r["failure_rate"] for r in summaries])
        axes[1].set(ylabel="Failure rate", ylim=(0, 1), title="Failure under random friction")
        fig.suptitle(
            "Uniform μ ∈ [0.2, 1.5], switches every 3 s\n"
            + "; ".join(
                f"{n}: {r['motions']} motions / {r['episodes']} episodes"
                for n, r in zip(names, summaries)
            )
            + (" — CENSORED PILOT" if m["max_seconds"] else "")
        )
        figure(fig, out / "01_random_friction_performance")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for variant in m["variants"]:
        curves = [np.mean(v, axis=0) for (vv, _), v in event_curves.items() if vv == variant]
        if curves:
            ax.plot(
                np.arange(50) * DT,
                np.mean(curves, axis=0),
                label=f"{variant}: {len(curves)} motions",
            )
    if ax.lines:
        ax.legend()
    ax.set(
        xlabel="Time after switch (s)",
        ylabel="Excess root position error vs own control (m)",
        title="Switch response: complete common windows, prefix/capacity gate passed",
    )
    ax.grid(alpha=0.2)
    figure(fig, out / "02_switch_response")
    if example:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
        axes[0].step(example["time"], example["mu"], where="post", label="Actual foot μ")
        axes[0].plot(example["time"], example["predicted_mu"], label="Encoder estimate")
        axes[0].set(
            ylabel="Friction μ", title=f"Illustrative final-policy trace: {example['name']}"
        )
        axes[0].legend()
        axes[1].plot(example["time"], example["position_error"])
        axes[1].set(ylabel="Root position error (m)")
        axes[2].plot(example["time"], example["action_change"])
        axes[2].set(xlabel="Time (s)", ylabel="Action change L2")
        figure(fig, out / "03_encoder_and_control")
    lines = [
        "# Exp009：随机摩擦切换实验",
        "",
        f"完成条件 {len(groups)}/{expected}；{'截断可行性验证，不是正式结果' if m['max_seconds'] else '完整动作评测'}。",
        "前3秒μ=1.0，此后每3秒从Uniform(0.2,1.5)独立采样。两模型共享每条动作与seed的预生成序列，失败后不重启。",
        "每个模型另采集一次不切换对照。预切换严格配对门限为1e-4；未通过者保留在总体鲁棒性统计，但排除机制性切换响应分析。",
        "",
        "| 模型 | 动作数 | 随机Episode数 | 平均回报 | 失败数 | 配对门限通过 | 容量受影响 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(
            f"| {r['variant']} | {r['motions']} | {r['episodes']} | {r['mean_return']:.4f} | {r['failures']} | {r['prefix_gate_passed']} | {r['capacity_affected']} |"
        )
    lines += [
        "",
        "配对效应（先按动作平均随机种子，再按动作bootstrap；不是按帧）：",
        "```json",
        json.dumps(effects, ensure_ascii=False, indent=2),
        "```",
        "",
        "control_adjusted_delta = (最终随机回报−最终对照回报)−(社区随机回报−社区对照回报)。正值表示最终模型的扰动损失较小。",
        "误差曲线仅用共同存活且完整的切换后1秒窗口，存在存活筛选；切换事件与随机种子均非独立动作样本。",
        "切换响应按动作等权，不能把动作阶段的变化解释为适应；没有定义恢复时间，因此不输出任意恢复时间。",
        "当前Encoder输出被限制在[0.3,1.2]。估计变化或动作变化不足以证明识别和适应；需要额外固定Encoder/历史消融。",
        "每步检查实际有约束的foot-plane接触，接触来自最后物理子步；空中无接触不被声称接触已验证。脚部滑动速度未实现。",
        "原始每步NPZ保留位置/姿态误差、动作、关节速度、base角速度、projected gravity、预测μ、接触数与有效μ、溢出及终止标记。",
    ]
    (out / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", required=True)
    report(output_path(p.parse_args().eval_dir))
