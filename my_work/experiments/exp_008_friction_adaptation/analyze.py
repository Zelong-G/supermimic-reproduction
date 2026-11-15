"""Descriptive paired analysis; no policy or world-model fitting."""

import csv
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run import HERE, OUT, save, sha


def table(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def data(stage, name, mu):
    with np.load(OUT / stage / f"{name}__mu{mu:g}.npz") as a:
        return {k: a[k] for k in a.files}


def wilson(k, n):
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def figure(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def latest(history):
    chunks = []
    start = 0
    for dim in [29, 29, 3, 3, 29]:
        chunks.append(history[:, start : start + 10 * dim].reshape(-1, 10, dim)[:, -1])
        start += 10 * dim
    return np.concatenate(chunks, axis=1)


def fixed(config):
    rows = []
    paired = []
    for case in config["cases"]:
        baseline = data("fixed", case["name"], 1.0)
        for mu in config["frictions"]:
            r = json.loads((OUT / "fixed" / f"{case['name']}__mu{mu:g}.json").read_text())
            a = data("fixed", case["name"], mu)
            assert np.max(abs(a["states"][0] - baseline["states"][0])) <= 1e-6
            assert np.max(abs(a["history"][0] - baseline["history"][0])) <= 1e-6
            assert np.allclose(a["mu"], mu)
            assert r["active_contact_samples"] > 0
            rows.append(
                {k: v for k, v in r.items() if k not in ["initial_state", "contact_evidence"]}
            )
            overlap = min(len(a["mu"]), len(baseline["mu"]))
            observed = latest(a["history"][:overlap])[25:]
            observed_base = latest(baseline["history"][:overlap])[25:]
            paired.append(
                {
                    "name": case["name"],
                    "category": case["category"],
                    "source": case["source"],
                    "mu": mu,
                    "common_steps": overlap,
                    "joint_position_difference_rms_rad": float(
                        np.sqrt(np.mean((observed[:, :29] - observed_base[:, :29]) ** 2))
                    ),
                    "joint_velocity_difference_rms_rad_s": float(
                        np.sqrt(np.mean((observed[:, 29:58] - observed_base[:, 29:58]) ** 2))
                    ),
                    "base_angular_velocity_difference_rms_rad_s": float(
                        np.sqrt(np.mean((observed[:, 58:61] - observed_base[:, 58:61]) ** 2))
                    ),
                    "projected_gravity_difference_rms": float(
                        np.sqrt(np.mean((observed[:, 61:64] - observed_base[:, 61:64]) ** 2))
                    ),
                    "previous_action_difference_rms": float(
                        np.sqrt(np.mean((observed[:, 64:] - observed_base[:, 64:]) ** 2))
                    ),
                    "score_delta_vs_mu1": float(
                        (a["reward"].sum() - baseline["reward"].sum()) / (0.1 * case["frames"])
                    ),
                    "mean_position_error_delta_common_prefix_m": float(
                        np.mean(
                            a["position_error"][:overlap] - baseline["position_error"][:overlap]
                        )
                    ),
                    "mean_orientation_error_delta_common_prefix_rad": float(
                        np.mean(
                            a["orientation_error"][:overlap]
                            - baseline["orientation_error"][:overlap]
                        )
                    ),
                }
            )
    table(OUT / "episodes.csv", rows)
    save(OUT / "episodes.json", rows)
    table(OUT / "paired_fixed_deltas.csv", paired)
    paired_map = {(r["name"], r["mu"]): r for r in paired}
    summaries = []
    categories = list(dict.fromkeys(c["category"] for c in config["cases"]))
    rng = np.random.default_rng(8)
    for category in ["All"] + categories:
        for mu in config["frictions"]:
            rs = [
                r
                for r in rows
                if r["mu"] == mu and (category == "All" or r["category"] == category)
            ]
            scores = np.array([r["norm_return"] for r in rs])
            n = len(rs)
            k = sum(r["failed"] for r in rs)
            # Stratum interval resamples the three source trajectories. Overall interval
            # clusters reused source trajectories across walking/turning strata.
            sources = sorted({r["source"] for r in rs})
            boot = []
            delta_boot = []
            for _ in range(2000):
                chosen = rng.choice(sources, len(sources), replace=True)
                boot.append(
                    np.mean(
                        [r["norm_return"] for source in chosen for r in rs if r["source"] == source]
                    )
                )
                delta_boot.append(
                    np.mean(
                        [
                            paired_map[r["name"], mu]["score_delta_vs_mu1"]
                            for source in chosen
                            for r in rs
                            if r["source"] == source
                        ]
                    )
                )
            lo, hi = np.quantile(boot, [0.025, 0.975])
            dlo, dhi = np.quantile(delta_boot, [0.025, 0.975])
            fl, fh = wilson(k, n)
            summaries.append(
                {
                    "category": category,
                    "mu": mu,
                    "motions": n,
                    "source_trajectories": len(sources),
                    "episodes": n,
                    "failures": k,
                    "failure_rate": k / n,
                    "failure_wilson_low": fl,
                    "failure_wilson_high": fh,
                    "mean_norm_return": float(scores.mean()),
                    "score_bootstrap_low": float(lo),
                    "score_bootstrap_high": float(hi),
                    "paired_mean_score_delta_vs_mu1": float(
                        np.mean([paired_map[r["name"], mu]["score_delta_vs_mu1"] for r in rs])
                    ),
                    "paired_delta_bootstrap_low": float(dlo),
                    "paired_delta_bootstrap_high": float(dhi),
                    "mean_root_position_error_m": float(
                        np.mean([r["mean_root_position_error_m"] for r in rs])
                    ),
                    "mean_root_orientation_error_rad": float(
                        np.mean([r["mean_root_orientation_error_rad"] for r in rs])
                    ),
                    "effective_steps": sum(r["steps"] for r in rs),
                    "overflow_steps": sum(r["overflow_steps"] for r in rs),
                    "capacity_overflow_steps": sum(r["capacity_overflow_steps"] for r in rs),
                    "solver_limit_steps": sum(r["solver_limit_steps"] for r in rs),
                }
            )
    table(OUT / "fixed_summary.csv", summaries)
    save(OUT / "fixed_summary.json", summaries)
    for metric, ylabel, filename in [
        ("mean_norm_return", "Mean normalized tracking return", "01_tracking_score"),
        ("failure_rate", "Failure rate", "02_failure_rate"),
    ]:
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 4.5), layout="constrained", gridspec_kw={"width_ratios": [1.7, 1]}
        )
        for category in categories:
            rs = sorted([r for r in summaries if r["category"] == category], key=lambda r: r["mu"])
            axes[0].plot(
                [r["mu"] for r in rs],
                [r[metric] for r in rs],
                "o-",
                label=f"{category} (3 clips / 3 episodes per μ)",
            )
        rs = sorted([r for r in summaries if r["category"] == "All"], key=lambda r: r["mu"])
        low = "score_bootstrap_low" if metric == "mean_norm_return" else "failure_wilson_low"
        high = "score_bootstrap_high" if metric == "mean_norm_return" else "failure_wilson_high"
        axes[1].plot([r["mu"] for r in rs], [r[metric] for r in rs], "ko-")
        axes[1].fill_between(
            [r["mu"] for r in rs],
            [r[low] for r in rs],
            [r[high] for r in rs],
            alpha=0.18,
            color="black",
        )
        axes[1].set_title(
            "All: 12 clips / 12 episodes per μ\n"
            + (
                "95% source-cluster bootstrap"
                if metric == "mean_norm_return"
                else "95% Wilson interval (descriptive)"
            )
        )
        for ax in axes:
            ax.axvspan(0.15, 0.3, color="orange", alpha=0.12)
            ax.set(
                xlabel="Effective foot–ground friction coefficient μ",
                ylabel=ylabel,
                xticks=sorted(config["frictions"]),
            )
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle("Frozen pretrained policy on selected 10 s LAFAN1 clips (60 episodes)")
        figure(fig, filename)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for category in categories:
        rs = sorted([r for r in summaries if r["category"] == category], key=lambda r: r["mu"])
        for ax, key, label in zip(
            axes,
            ["mean_root_position_error_m", "mean_root_orientation_error_rad"],
            ["Root position error (m)", "Root orientation error (rad)"],
        ):
            ax.plot([r["mu"] for r in rs], [r[key] for r in rs], "o-", label=category)
            ax.set(xlabel="Effective foot–ground friction coefficient μ", ylabel=label)
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Pelvis tracking errors, conditional on observed pre-failure states\n3 selected clips / 3 episodes per stratum per μ; durations differ after failure"
    )
    figure(fig, "06_fixed_root_errors")
    return rows, summaries


def pairing_audit(config):
    rows = []
    for case in [c for c in config["cases"] if c["category"] == "Walking"]:
        base = data("switch", case["name"], 1.0)
        with np.load(OUT / "switch" / f"{case['name']}__mu1_repeat.npz") as a:
            repeat = {k: a[k] for k in a.files}
        prefix = {}
        for label, a in [
            ("control_repeat", repeat),
            ("switch_0.3", data("switch", case["name"], 0.3)),
            ("switch_0.15", data("switch", case["name"], 0.15)),
        ]:
            prefix[label] = {
                key: float(np.max(abs(a[key][:150] - base[key][:150])))
                for key in ["states", "history", "goal", "actions", "reward"]
            }
        eligible = max(v for d in prefix.values() for v in d.values()) <= 1e-4
        rows.append(
            {
                "name": case["name"],
                "eligible": eligible,
                "prefix_differences": prefix,
                "reason": "passes predeclared 1e-4 absolute prefix tolerance"
                if eligible
                else "control repeat or intervention prefix exceeds predeclared tolerance; excluded from online-response inference",
            }
        )
    result = {
        "tolerance": 1e-4,
        "attempted_clips": len(rows),
        "eligible_clips": [r["name"] for r in rows if r["eligible"]],
        "excluded_clips": [r["name"] for r in rows if not r["eligible"]],
        "details": rows,
    }
    save(OUT / "pairing_audit.json", result)
    return result


def switches(config):
    rows = []
    times = []
    windows = []
    audit = pairing_audit(config)
    cases = [c for c in config["cases"] if c["name"] in audit["eligible_clips"]]
    assert cases, "No switch case passes the pairing gate"
    fig, axes = plt.subplots(
        2, len(cases), figsize=(max(7, 5 * len(cases)), 7), layout="constrained", squeeze=False
    )
    response_fig, response_axes = plt.subplots(4, 2, figsize=(12, 11), layout="constrained")
    step = config["switch_step"]
    for col, case in enumerate(cases):
        base = data("switch", case["name"], 1.0)
        with np.load(OUT / "switch" / f"{case['name']}__mu1_repeat.npz") as loaded:
            repeat = {k: loaded[k] for k in loaded.files}
        repeat_floor = {
            k: float(np.max(abs(repeat[k][:step] - base[k][:step])))
            for k in ["states", "history", "goal", "actions", "reward"]
        }
        if max(repeat_floor.values()) > 1e-4:
            raise ValueError(f"Numerical control prefix is not reproducible: {repeat_floor}")
        assert not base["terminated"].any(), "Part B control must be stable"
        for mu in [1.0, 0.3, 0.15]:
            a = data("switch", case["name"], mu)
            n = min(len(a["mu"]), len(base["mu"]))
            t = np.arange(n) * 0.02
            diffs = {
                k: float(np.max(abs(a[k][:step] - base[k][:step])))
                for k in ["states", "history", "goal", "actions", "reward"]
            }
            if max(diffs.values()) > 1e-4:
                raise ValueError(f"Unpaired prefix: {case['name']}, {mu}: {diffs}")
            assert np.allclose(a["mu"][:step], 1.0) and np.allclose(a["mu"][step:], mu)
            dp = a["position_error"][:n] - base["position_error"][:n]
            dq = a["orientation_error"][:n] - base["orientation_error"][:n]
            disrupted = np.flatnonzero((dp[step:] > 0.05) | (dq[step:] > 0.10)) + step
            recovery = None
            status = "no_detected_disruption"
            if len(disrupted):
                status = "not_recovered"
                for k in range(int(disrupted[0]) + 1, n - 49):
                    if np.all(dp[k : k + 50] <= 0.05) and np.all(dq[k : k + 50] <= 0.10):
                        recovery = (k - step) * 0.02
                        status = "recovered"
                        break
            row = {
                "name": case["name"],
                "mu": mu,
                "steps": n,
                "failed": bool(a["terminated"].any()),
                "pre_switch_max_abs_difference": max(diffs.values()),
                "pre_switch_differences": diffs,
                "control_repeat_prefix_max_difference": max(repeat_floor.values()),
                "control_repeat_post_position_max_difference_m": float(
                    np.max(abs(repeat["position_error"][step:] - base["position_error"][step:]))
                ),
                "control_repeat_post_orientation_max_difference_rad": float(
                    np.max(
                        abs(repeat["orientation_error"][step:] - base["orientation_error"][step:])
                    )
                ),
                "control_repeat_post_action_rms_difference": float(
                    np.sqrt(np.mean((repeat["actions"][step:] - base["actions"][step:]) ** 2))
                ),
                "post_excess_position_mean_m": float(dp[step:].mean()),
                "post_excess_orientation_mean_rad": float(dq[step:].mean()),
                "post_excess_position_max_m": float(dp[step:].max()),
                "post_excess_orientation_max_rad": float(dq[step:].max()),
                "recovery_status": status,
                "recovery_time_s": recovery,
                "overflow_steps": int(a["overflow"].sum()),
            }
            rows.append(row)
            current = latest(a["history"][:n])
            base_current = latest(base["history"][:n])
            for label, start, stop in [
                ("pre", 1.0, 3.0),
                ("early_post", 3.0, 5.0),
                ("late_post", 5.0, 10.0),
            ]:
                valid = (t >= start) & (t < stop)
                if not valid.any():
                    continue
                windows.append(
                    {
                        "name": case["name"],
                        "mu": mu,
                        "window": label,
                        "start_s": start,
                        "stop_s": stop,
                        "observed_steps": int(valid.sum()),
                        "planned_steps": round((stop - start) / 0.02),
                        "mean_position_error_m": float(a["position_error"][:n][valid].mean()),
                        "mean_orientation_error_rad": float(
                            a["orientation_error"][:n][valid].mean()
                        ),
                        "mean_excess_position_m": float(dp[valid].mean()),
                        "mean_excess_orientation_rad": float(dq[valid].mean()),
                        "mean_reward": float(a["reward"][:n][valid].mean()),
                        "joint_velocity_rms": float(
                            np.sqrt(np.mean(a["states"][:n, 42:][valid] ** 2))
                        ),
                        "paired_action_difference_rms": float(
                            np.sqrt(
                                np.mean((a["actions"][:n][valid] - base["actions"][:n][valid]) ** 2)
                            )
                        ),
                        "paired_joint_velocity_difference_rms": float(
                            np.sqrt(
                                np.mean((current[valid, 29:58] - base_current[valid, 29:58]) ** 2)
                            )
                        ),
                        "paired_base_angular_velocity_difference_rms": float(
                            np.sqrt(
                                np.mean((current[valid, 58:61] - base_current[valid, 58:61]) ** 2)
                            )
                        ),
                        "paired_projected_gravity_difference_rms": float(
                            np.sqrt(
                                np.mean((current[valid, 61:64] - base_current[valid, 61:64]) ** 2)
                            )
                        ),
                        "mean_active_foot_contacts": float(
                            a["active_foot_contacts"][:n][valid].mean()
                        ),
                    }
                )
            for i in range(n):
                times.append(
                    {
                        "name": case["name"],
                        "mu": mu,
                        "time_s": float(t[i]),
                        "position_error_m": float(a["position_error"][i]),
                        "orientation_error_rad": float(a["orientation_error"][i]),
                        "reward": float(a["reward"][i]),
                        "action_delta_rms": float(
                            np.sqrt(np.mean((a["actions"][i] - base["actions"][i]) ** 2))
                        ),
                        "joint_velocity_rms": float(np.sqrt(np.mean(a["states"][i, 42:] ** 2))),
                        "active_foot_contacts": int(a["active_foot_contacts"][i]),
                        "effective_mu": float(a["mu"][i]),
                    }
                )
            for ax, key in zip(axes[:, col], ["position_error", "orientation_error"]):
                ax.plot(t, a[key][:n], label=f"1.0 → {mu:g}")
            if col == 0 and mu != 1.0:
                current = latest(a["history"][:n])
                bc = latest(base["history"][:n])
                values = [
                    np.sqrt(np.mean((a["actions"][:n] - base["actions"][:n]) ** 2, axis=1)),
                    np.sqrt(np.mean((current[:, 29:58] - bc[:, 29:58]) ** 2, axis=1)),
                    np.sqrt(np.mean((current[:, 58:61] - bc[:, 58:61]) ** 2, axis=1)),
                    np.sqrt(np.mean((current[:, 61:64] - bc[:, 61:64]) ** 2, axis=1)),
                ]
                labels = [
                    "Policy action difference (RMS)",
                    "Joint velocity difference (rad/s RMS)",
                    "Base angular velocity difference (rad/s RMS)",
                    "Projected gravity difference (RMS)",
                ]
                for r, (v, label) in enumerate(zip(values, labels)):
                    ax = response_axes[r, 0 if mu == 0.3 else 1]
                    ax.plot(t, v)
                    ax.axvline(3, color="k", ls="--")
                    ax.set(ylabel=label, xlabel="Time (s)", title=f"Paired 1.0 → {mu:g} vs control")
                    ax.grid(alpha=0.2)
        axes[0, col].set_title(case["name"].replace("__", "\n"))
        for r, ax in enumerate(axes[:, col]):
            ax.axvline(3, color="k", ls="--")
            ax.set(
                xlabel="Time (s)",
                ylabel="Root position error (m)" if r == 0 else "Root orientation error (rad)",
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Sudden friction change at 3 s: {len(cases)} eligible clip, {3 * len(cases)} episodes + {len(cases)} numerical control\n3 clips attempted; strict 1e-4 prefix gate; history retained"
    )
    figure(fig, "03_sudden_change_errors")
    response_fig.suptitle(
        f"Policy and observation response: {cases[0]['name']}\nPaired differences demonstrate feedback response, not identification of μ"
    )
    figure(response_fig, "04_policy_observation_changes")
    save(OUT / "switch_summary.json", rows)
    table(
        OUT / "switch_summary.csv",
        [{k: v for k, v in r.items() if k != "pre_switch_differences"} for r in rows],
    )
    table(OUT / "switch_timeseries.csv", times)
    table(OUT / "switch_window_summary.csv", windows)
    return rows


def inference(config):
    # All conditions of a source motion stay together. Fixed, predeclared ridge
    # strength; no hyperparameter selection using held-out data.
    xs = []
    ys = []
    groups = []
    episodes = []
    for ci, case in enumerate(config["cases"]):
        arrays = [data("fixed", case["name"], mu) for mu in config["frictions"]]
        stop = min(len(a["mu"]) for a in arrays)
        ids = np.arange(25, stop, 5)  # t>=0.5 s; common times across friction, 10 Hz
        for mu, a in zip(config["frictions"], arrays):
            xs.extend(a["history"][ids])
            ys.extend([mu] * len(ids))
            groups.extend([case["source"]] * len(ids))
            episodes.extend([f"{ci}_{mu}"] * len(ids))
    x = np.array(xs, dtype=float)
    y = np.array(ys)
    g = np.array(groups)
    e = np.array(episodes)
    unique = sorted(set(groups))
    foldmap = {s: i % 3 for i, s in enumerate(unique)}
    fold = np.array([foldmap[s] for s in g])
    results = []
    predictions = []
    for label, features in [("Single frame", latest(x)), ("10-frame history", x)]:
        pred = np.zeros(len(y))
        baseline = np.zeros(len(y))
        for f in range(3):
            tr = fold != f
            te = fold == f
            assert not set(g[tr]) & set(g[te])
            mean = features[tr].mean(0)
            std = features[tr].std(0)
            std = np.maximum(std, 1e-4)
            a = (features[tr] - mean) / std
            b = (features[te] - mean) / std
            ym = y[tr].mean()
            target = y[tr] - ym
            # Solve the smaller ridge system, alpha=100, using training-only scaling.
            if a.shape[0] >= a.shape[1]:
                coef = np.linalg.solve(a.T @ a + 100 * np.eye(a.shape[1]), a.T @ target)
            else:
                coef = a.T @ np.linalg.solve(a @ a.T + 100 * np.eye(a.shape[0]), target)
            pred[te] = b @ coef + ym
            baseline[te] = ym
        # Every episode receives equal metric weight, despite different failure times.
        mae = np.mean([np.mean(abs(pred[e == ep] - y[e == ep])) for ep in set(e)])
        mse = np.mean([np.mean((pred[e == ep] - y[e == ep]) ** 2) for ep in set(e)])
        bmse = np.mean([np.mean((baseline[e == ep] - y[e == ep]) ** 2) for ep in set(e)])
        source_mae = {s: float(np.mean(abs(pred[g == s] - y[g == s]))) for s in unique}
        rng = np.random.default_rng(8)
        boot = [np.mean(rng.choice(list(source_mae.values()), len(unique))) for _ in range(2000)]
        lo, hi = np.quantile(boot, [0.025, 0.975])
        results.append(
            {
                "features": label,
                "mae": float(mae),
                "rmse": float(np.sqrt(mse)),
                "r2_vs_train_mean": float(1 - mse / bmse),
                "source_mean_mae": float(np.mean(list(source_mae.values()))),
                "source_bootstrap_low": float(lo),
                "source_bootstrap_high": float(hi),
                "source_mae": source_mae,
                "samples": len(y),
                "episodes": len(set(e)),
                "source_groups": len(unique),
                "folds": 3,
                "alpha": 100,
            }
        )
        predictions.extend(
            {
                "features": label,
                "source": str(s),
                "episode": str(ep),
                "fold": int(f),
                "true_mu": float(t),
                "predicted_mu": float(p),
            }
            for s, ep, f, t, p in zip(g, e, fold, y, pred)
        )
    baseline_mae = float(np.mean([np.mean(abs(baseline[e == ep] - y[e == ep])) for ep in set(e)]))
    baseline_source_mae = {s: float(np.mean(abs(baseline[g == s] - y[g == s]))) for s in unique}
    baseline_row = {
        **results[0],
        "features": "Training-mean baseline",
        "mae": baseline_mae,
        "rmse": float(np.sqrt(bmse)),
        "r2_vs_train_mean": 0.0,
        "source_mean_mae": float(np.mean(list(baseline_source_mae.values()))),
        "source_mae": baseline_source_mae,
        "source_bootstrap_low": float(min(baseline_source_mae.values())),
        "source_bootstrap_high": float(max(baseline_source_mae.values())),
        "alpha": None,
    }
    save(
        OUT / "history_inference.json",
        {
            "results": results,
            "baseline": baseline_row,
            "fold_assignment": foldmap,
            "protocol": "Source-motion grouped 3-fold cross-validation; all friction conditions and segments from a source remain in one fold. Common surviving time prefix across all 5 frictions per clip; stride 5, omit first 0.5 s; training-only standardization; ridge alpha=100 fixed; no clipping. Exploratory information probe, not control improvement or causal identification.",
        },
    )
    table(
        OUT / "history_inference.csv",
        [{k: v for k, v in r.items() if k != "source_mae"} for r in results + [baseline_row]],
    )
    table(OUT / "history_predictions.csv", predictions)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for i, r in enumerate(results):
        axes[0].bar(i, r["source_mean_mae"], color=["#4076a9", "#df9037"][i])
        axes[0].errorbar(
            i,
            r["source_mean_mae"],
            yerr=[
                [r["source_mean_mae"] - r["source_bootstrap_low"]],
                [r["source_bootstrap_high"] - r["source_mean_mae"]],
            ],
            fmt="k_",
            capsize=5,
        )
        ps = [p for p in predictions if p["features"] == r["features"]]
        means = [
            np.mean([p["predicted_mu"] for p in ps if p["true_mu"] == mu])
            for mu in sorted(config["frictions"])
        ]
        axes[1].plot(sorted(config["frictions"]), means, "o-", label=r["features"])
    axes[0].axhline(
        baseline_row["source_mean_mae"], color="black", ls="--", label="Training-mean baseline"
    )
    axes[0].legend(fontsize=8)
    axes[1].axhline(float(baseline.mean()), color="gray", ls=":", label="Training-mean baseline")
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Single frame", "10-frame history"],
        ylabel="Source-averaged held-out MAE of μ",
        title="95% source-cluster bootstrap intervals",
    )
    axes[1].plot([0.15, 1], [0.15, 1], "k--", label="Identity")
    axes[1].set(xlabel="True friction coefficient μ", ylabel="Mean held-out prediction")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        f"Exploratory friction inference: {len(unique)} source groups, {len(set(e))} episodes\nSource-grouped 3-fold CV; shared surviving time prefixes; frozen policy"
    )
    figure(fig, "05_history_friction_inference")
    return results + [baseline_row]


def analyze():
    config = json.loads((HERE / "config.json").read_text())
    (OUT / "figures").mkdir(exist_ok=True)
    _rows, summary = fixed(config)
    switch = switches(config)
    audit = json.loads((OUT / "pairing_audit.json").read_text())
    probe = inference(config)
    save(
        OUT / "analysis_provenance.json",
        {
            "config_sha256": sha(HERE / "config.json"),
            "code_sha256": {p.name: sha(p) for p in HERE.glob("*.py")},
            "trajectory_sha256": {
                str(p.relative_to(OUT)): sha(p)
                for d in ["fixed", "switch"]
                for p in (OUT / d).glob("*.npz")
            },
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    )
    lines = [
        "# Experiment 008: Friction Adaptation on Unseen LAFAN1 Motions",
        "",
        'Frozen community checkpoint; selected existing 10-second LAFAN1 segments. No BONES-SEED, downloads, policy training or world-model training. "Unseen" follows the prior LAFAN1 evaluation protocol; the exact pretraining corpus membership cannot independently be proven from the checkpoint.',
        "",
        "## Fixed friction results",
        "",
        "| μ | Return | Failures / episodes | Root position error (m) | Root orientation error (rad) | Valid steps | Capacity flags | Solver-limit flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r["category"] == "All":
            lines.append(
                f"| {r['mu']} | {r['mean_norm_return']:.4f} | {r['failures']}/{r['episodes']} | {r['mean_root_position_error_m']:.4f} | {r['mean_root_orientation_error_rad']:.4f} | {r['effective_steps']} | {r['capacity_overflow_steps']} | {r['solver_limit_steps']} |"
            )
    lines += [
        "",
        "Per-category n=3 episodes per μ; see fixed_summary.csv for failure counts and uncertainty. One deterministic seed; rerunning identical zero-noise seeds would not add independent evidence. Selected sources overlap between walking and turning; overall bootstrap clusters by source. Wilson failure intervals are descriptive and do not account for this overlap.",
        "",
        "## Actual motions",
        "",
        "| Stratum | Existing segment | Parent motion | Reference yaw range (deg) | Historical score |",
        "|---|---|---|---:|---:|",
    ]
    for c in config["cases"]:
        lines.append(
            f"| {c['category']} | {c['name']} | {c['source']} | {c['yaw_range_deg']:.1f} | {c['historical_score']:.4f} |"
        )
    lines += [
        "",
        "Turning is a quantitative walking sub-stratum (pelvis unwrapped yaw range >=45 degrees), not a new LAFAN1 action label. Turning examples are the largest yaw ranges among historically successful, overflow-free segments, with distinct parents. Other strata use preselected source clips from Exp003, not exhaustive or randomly sampled categories. Walking can also include turns.",
        "",
        "## Sudden change",
        "",
        f"Attempted 3 clips / 12 episodes including numerical repeats. Eligible clips: {audit['eligible_clips']}. Excluded clips: {audit['excluded_clips']}; their control-repeat or switched prefixes exceeded the predeclared 1e-4 absolute tolerance before friction changed. Full data and per-term differences remain in pairing_audit.json. Main switch figures/statistics use eligible clips only; the exclusion is a numerical reproducibility limitation, not evidence of policy friction failure.",
        "",
        "Switch writes the existing Warp buffer at step 150, before integrating [3.00, 3.02) s. No episode reset, history reset, robot reinitialization, or checkpoint reload occurs at the switch. All prefixes are checked for state, history, goal, action and reward agreement within 1e-4 absolute tolerance, covering small floating-point velocity differences (~5e-5) in pilot controls. Three extra unchanged-control repeats measure the numerical floor; they are not independent behavioral samples. Actual simulator times and buffer pointers are checked. Signals plotted at pre-action timestamps; reward and contacts are from the ensuing control interval. Contacts are sampled at the last physics substep only.",
        "",
        "Recovery criterion, declared before analysis: " + config["recovery"],
        "",
        "| Clip | Target μ | Prefix max difference | Mean excess position (m) | Mean excess orientation (rad) | Failed | Recovery | Time (s) |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for r in switch:
        lines.append(
            f"| {r['name']} | {r['mu']} | {r['pre_switch_max_abs_difference']:.3g} | {r['post_excess_position_mean_m']:.4f} | {r['post_excess_orientation_mean_rad']:.4f} | {r['failed']} | {r['recovery_status']} | {r['recovery_time_s']} |"
        )
    lines += [
        "",
        "## History information probe",
        "",
        "| Features | Episode-weighted MAE | RMSE | R² vs training mean |",
        "|---|---:|---:|---:|",
    ]
    for r in probe:
        lines.append(
            f"| {r['features']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['r2_vs_train_mean']:.4f} |"
        )
    lines += [
        "",
        "Exploratory diagnostic ridge regression only. Source-grouped folds prevent neighboring frames and conditions of the same source crossing train/test. All five μ values appear in training: this is not inference on unseen friction labels. The probe may exploit motion-dependent feedback responses and action correlations. It does not establish that the policy infers μ or uses its history beneficially. Small source count and common-prefix survivor selection limit generalization. Contact excitation is necessary; low-dynamic windows can be uninformative.",
        "",
        "## Interface and contact audit",
        "",
        "- Actual checkpoint policy: history=10, observation=930 (29 joint pos +29 joint vel +3 base angular vel +3 projected gravity +29 previous actions, each ten frames), goal=650 (10×59 SONIC reference features +60 pelvis_ori10 features), running normalization and ELU. No ξ input to policy; world model context is separate.",
        "- Physics step 0.005 s, decimation 4, control step 0.02 s. Ten frames span 0.18 s from oldest to newest; 0.20 s when counted as ten sampling intervals. Reset history is backfilled.",
        "- Training reset DR: absolute shared tangential foot friction U[0.3,1.2], 14 foot geoms. Evaluation startup fixes those geoms at 1.0. Ground μ=1.0, priority 0; feet priority 1, condim=3. No explicit contact pairs.",
        "- Geom metadata stores compiled CPU defaults (feet initially 0.6); startup events set the live Warp feet to 1.0. Episode mu and contact friction are the live measurements, not the CPU defaults.",
        "- Formal solver iterations=100, line-search iterations=80, uniformly across friction conditions (community defaults 10/20). Small pilots establish feasibility and numerical sensitivity; incomplete iter10 exploration is excluded. For any capacity-affected source segment, ALL five conditions are rerun with nconmax=140, njmax=1000; initial records remain in capacity_affected_initial. fixed/repair_metadata.json lists these paired replacements. Other segments retain 35/250; Part B uses 140/1000 throughout. These are storage/solver changes, not changed material parameters; validation must confirm zero capacity errors in final data.",
        "- In-place writes to the already-expanded geom_friction array preserve CUDA graph references. Every active sampled foot-ground contact is checked against requested μ. contact_probe.json and per-episode contact_evidence retain full 5D contact friction plus constraint addresses.",
        "- Only foot tangential friction changes; torsional/rolling coefficients and all other geoms stay fixed. This matches training ξ semantics. In fall/get-up, hands/torso ground contacts remain at their original friction; this experiment is not a uniform slippery-floor material change.",
        "- MuJoCo priority rule: https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters ; installed Warp implementation: collision_core.contact_material_params.",
        "- All noises and unrelated DR disabled, paired reference-frame initial states. Returns use full segment denominator 0.1×500, including early failure penalty. Errors are pelvis world-pose Euclidean/geodesic errors before action (not torso reward anchors), averaged over surviving states. Time to failure is null for censored completions.",
        "- Raw states/actions/observations/velocities/contacts/rewards retained in NPZ; episode CSV/JSON and switch time-series CSV preserve statistics. Reliable contact-point sliding velocity was not implemented; contact count alone is not slip.",
        "",
        "## Interpretation limits",
        "",
        "Friction robustness measures performance across fixed conditions. Online feedback response after a change is compatible with adaptation but not proof of learned dynamics identification. Implicit inference concerns information in observations; a diagnostic probe does not prove its causal use by the controller. μ=0.15 is outside the configured friction range, but checkpoint training exposure and all other dimensions are not independently audited. Failure cannot be attributed to missing ξ input. Causal evidence for history improving control requires matched history ablation/control experiments, which are outside this first behavioral study. Ten-second segment success does not imply full-clip success. No network changes are motivated by these results alone.",
        "",
        "## Figures",
        "",
        "1. figures/01_tracking_score.png / .pdf",
        "2. figures/02_failure_rate.png / .pdf",
        "3. figures/03_sudden_change_errors.png / .pdf",
        "4. figures/04_policy_observation_changes.png / .pdf",
        "5. figures/05_history_friction_inference.png / .pdf",
        "6. figures/06_fixed_root_errors.png / .pdf",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "fixed": [r for r in summary if r["category"] == "All"],
                "switch": switch,
                "inference": probe,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    analyze()
