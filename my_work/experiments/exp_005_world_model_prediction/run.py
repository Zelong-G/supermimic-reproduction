"""World Model feasibility, short diagnostic captures and frozen-model error analysis."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def config():
    return json.loads((HERE / "config.json").read_text())


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feasibility(cfg):
    import numpy as np
    import torch

    from realmimic.export.web_world_model import load_world_model

    torch.set_num_threads(2)
    wm, ck = load_world_model(ROOT / cfg["checkpoint"])
    artifacts = []
    for case in cfg["cases"]:
        path = (
            ROOT
            / "my_work/outputs/exp_004_motion_visualization/recordings/capture_001"
            / case["motion"]
            / "raw/trajectory.npz"
        )
        with np.load(path) as data:
            artifacts.append(
                {
                    "motion": case["motion"],
                    "exp004_arrays": {k: list(data[k].shape) for k in data.files},
                }
            )
    report = {
        "status": "feasible_with_new_short_transition_capture",
        "checkpoint": cfg["checkpoint"],
        "checkpoint_sha256": digest(ROOT / cfg["checkpoint"]),
        "checkpoint_keys": list(ck),
        "wm_tensors": len(ck["wm"]),
        "strict_load_succeeded": True,
        "iteration": ck["iteration"],
        "world_model_config": ck["config"]["world_model"],
        "interface": "WorldModel.step(state[...,71], executed_action[...,29], xi[...,33]) -> next_state[...,71]",
        "control_dt_s": wm.spec.dt,
        "feature_frame": wm.spec.frame,
        "exp004_artifacts": artifacts,
        "obstacles": [
            "Exp004 does not store qvel, executed actions or xi. No valid direct reuse or finite-difference substitution.",
            "The model predicts generalized state, not contact/solver caches or actuator internal states.",
            "Capacity overflow windows must be identified. Diagnostic trajectories do not repeat the full benchmark.",
        ],
        "protocol": "At most 512 steps per selected clip, from the reference start; stop at first done. Same logged executed actions for teacher forcing and autoregression.",
        "sources": [
            "src/supermimic/world_model/world_model.py",
            "src/supermimic/world_model/state_spec.py",
            "src/realmimic/export/web_world_model.py",
            "src/supermimic/envs/state_io.py",
            "src/supermimic/envs/actions.py",
            "src/supermimic/envs/sim_context.py",
        ],
    }
    save_json(ROOT / cfg["output_dir"] / "feasibility.json", report)
    print(
        "Strict pretrained WM loading succeeded; Exp004 lacks qvel/actions/xi; short recapture required."
    )
    return report


def capture(cfg, device):
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.envs.sim_context import SimContext
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX
    from supermimic.eval.benchmark import env_kwargs, load_policy_from_checkpoint

    torch.set_num_threads(2)
    policy, checkpoint = load_policy_from_checkpoint(ROOT / cfg["checkpoint"], device)
    configure_torch_backends()
    for case in cfg["cases"]:
        out = ROOT / cfg["output_dir"] / case["motion"]
        raw = out / "raw"
        if (raw / "transitions.npz").exists():
            raise FileExistsError(
                f"{raw}/transitions.npz already exists; use analyze without recapture"
            )
        env_cfg = make_env_cfg(
            SuperMimicEnvCfgArgs(
                npz_files=[str(ROOT / cfg["npz_dir"] / f"{case['motion']}.npz")],
                num_envs=1,
                episode_length_s=1e5,
                action_noise_std=0,
                obs_noise=False,
                reset_noise=False,
                domain_randomization=False,
                sampling_mode="start",
                record_transitions=False,
                seed=0,
                actuator_gains="spec",
                **env_kwargs(checkpoint),
            )
        )
        # Retain the true terminal qpos/qvel/action instead of reading an auto-reset state.
        env_cfg.auto_reset = False
        env = ManagerBasedRlEnv(env_cfg, device=device)
        try:
            if not np.isclose(env.step_dt, checkpoint["world_model"]["state_spec"]["dt"]):
                raise ValueError("Environment/model timestep mismatch")
            cmd = env.command_manager.get_term("motion")
            cmd.forced_frame[:] = cmd.library.clip_start[0]
            obs, _ = env.reset()
            io = StateIO(env)
            context = SimContext(env, FOOT_GEOM_REGEX)
            s0 = io.read()[0].cpu().numpy()
            states = [s0]
            actions = []
            clean_actions = []
            contexts = []
            terminated = []
            truncated = []
            overflows = []
            for _ in range(cfg["max_steps"]):
                contexts.append(context.read()[0].cpu().numpy())
                with torch.inference_mode():
                    action = policy(obs["policy"], obs["goal"])
                obs, _, term, trunc, _ = env.step(action)
                executed = env.action_manager.get_term("joint_pos").executed_raw[0].clone()
                clean_actions.append(action[0].cpu().numpy())
                actions.append(executed.cpu().numpy())
                states.append(io.read()[0].cpu().numpy())
                terminated.append(bool(term[0]))
                truncated.append(bool(trunc[0]))
                overflow = getattr(env.sim.data, "overflow", None)
                overflows.append(False if overflow is None else bool((overflow[0] != 0).item()))
                if overflow is not None:
                    overflow[:] = 0
                if terminated[-1] or truncated[-1]:
                    break
            arrays = {
                "states": np.asarray(states),
                "actions": np.asarray(actions),
                "policy_actions": np.asarray(clean_actions),
                "xi": np.asarray(contexts),
                "terminated": np.asarray(terminated),
                "truncated": np.asarray(truncated),
                "overflow": np.asarray(overflows),
                "dt": np.array(env.step_dt),
            }
            from analysis import validate_trajectory

            validate_trajectory(arrays)
            if not np.allclose(arrays["actions"], arrays["policy_actions"]):
                raise ValueError("Unexpected action processing mismatch in noise-free capture")
            raw.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raw / "transitions.npz", **arrays)
            old_path = (
                ROOT
                / "my_work/outputs/exp_004_motion_visualization/recordings/capture_001"
                / case["motion"]
                / "summary.json"
            )
            old = json.loads(old_path.read_text())
            with np.load(old_path.parent / "raw/trajectory.npz") as prior:
                overlap = min(len(prior["qpos"]), len(states) - 1)
                old_q = prior["qpos"][:overlap]
                qdelta = np.max(np.abs(arrays["states"][:overlap, :36] - old_q), axis=1)
            meta = {
                "motion": case["motion"],
                "label": case["label"],
                "group": case["group"],
                "transitions": len(actions),
                "duration_s": len(actions) * env.step_dt,
                "stop_reason": "termination"
                if terminated[-1]
                else "clip_end"
                if truncated[-1]
                else "diagnostic_cap",
                "overflow_transitions": sum(overflows),
                "terminal_state_retained": True,
                "auto_reset": False,
                "control_dt_s": env.step_dt,
                "action_units": "executed_raw normalized PD position command; not torque",
                "xi_description": "measured friction, 29 encoder biases, 3 torso COM offsets before each step",
                "exp004_overlap_max_abs_qpos_difference": float(qdelta.max()),
                "exp004_first_difference_over_1e-5_step": next(
                    (int(i) for i, e in enumerate(qdelta) if e > 1e-5), None
                ),
                "exp004_failed": old["result"]["failed"],
                "exp004_fail_step": old["result"]["fail_step"],
                "checkpoint_sha256": digest(ROOT / cfg["checkpoint"]),
                "config_sha256": digest(HERE / "config.json"),
                "trajectory_sha256": digest(raw / "transitions.npz"),
                "state_description": "origin-free world position, wxyz quaternion, joint pos, world linear vel, body angular vel, joint vel",
                "physics_dt_s": env.physics_dt,
                "decimation": env.cfg.decimation,
            }
            save_json(out / "capture.json", meta)
            print(
                "Captured",
                case["motion"],
                len(actions),
                "transitions;",
                meta["stop_reason"],
                "overflow",
                sum(overflows),
                flush=True,
            )
        finally:
            env.close()
            del env
            gc.collect()
            torch.cuda.empty_cache()


def analyze(cfg, device):
    import numpy as np
    import torch
    from analysis import evaluate

    from realmimic.export.web_world_model import load_world_model
    from supermimic.world_model.state_spec import StateSpec

    torch.set_num_threads(2)
    wm, _ = load_world_model(ROOT / cfg["checkpoint"])
    wm.to(device)
    wm.spec = StateSpec(wm.cfg.state_spec, device)
    all_rows = []
    for case in cfg["cases"]:
        out = ROOT / cfg["output_dir"] / case["motion"]
        with np.load(out / "raw/transitions.npz") as loaded:
            data = {k: loaded[k] for k in loaded.files}
        rows, detail = evaluate(
            wm, data, cfg["horizons"], cfg["window_stride"], cfg["max_windows"], device
        )
        np.savez_compressed(out / "raw/prediction_errors.npz", **detail)
        all_rows += [{**case, **r} for r in rows]
    output = ROOT / cfg["output_dir"]
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "config": cfg,
        "rows": all_rows,
        "versions": {
            p: importlib.metadata.version(p) for p in ["torch", "numpy", "matplotlib", "mujoco"]
        },
        "code_sha256": {p.name: digest(p) for p in HERE.glob("*.py")},
        "interpretation": "Dynamics prediction on the same recorded executed actions; no policy reevaluation in predicted states. 71 generalized coordinates/velocities do not include full solver internal state.",
    }
    save_json(output / "summary.json", summary)
    figures(cfg, all_rows, output)
    report(cfg, all_rows, output)
    print("Analysis and English figures saved:", output, flush=True)


def figures(cfg, rows, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    choices = [
        ("normalized_state_mse", "Normalized state MSE (70-D)"),
        ("root_position_rmse_m", "Root position RMSE (m)"),
        ("root_orientation_rmse_rad", "Root orientation RMSE (rad)"),
        ("joint_position_rmse_rad", "Joint position RMSE (rad)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, (key, label) in zip(axes.flat, choices):
        for case in cfg["cases"]:
            selected = [
                r
                for r in rows
                if r["motion"] == case["motion"]
                and r["protocol"] == "autoregressive"
                and r["subset"] == "no_overflow_32"
            ]
            ax.plot(
                [r["horizon"] for r in selected],
                [r[key] for r in selected],
                "o-",
                label=case["label"],
            )
        ax.set(
            xlabel="Autoregressive horizon (control steps; 20 ms each)",
            ylabel=label,
            xticks=cfg["horizons"],
        )
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("World Model vs MuJoCo: identical logged actions, overflow-free windows")
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"01_rollout_errors.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    import numpy as np

    for i, case in enumerate(cfg["cases"]):
        with np.load(out / case["motion"] / "raw/prediction_errors.npz") as d:
            axes[0].plot(
                np.arange(len(d["teacher_normalized_state_mse"])) * 0.02,
                d["teacher_normalized_state_mse"],
                label=case["label"],
                alpha=0.8,
            )
        teacher = next(
            r
            for r in rows
            if r["motion"] == case["motion"]
            and r["protocol"] == "teacher_forced"
            and r["subset"] == "no_overflow_transition"
        )
        ar = next(
            r
            for r in rows
            if r["motion"] == case["motion"]
            and r["protocol"] == "autoregressive"
            and r["subset"] == "no_overflow_32"
            and r["horizon"] == 32
        )
        bars = axes[1].bar(
            [i * 3, i * 3 + 1],
            [teacher["normalized_state_mse"], ar["normalized_state_mse"]],
            color=["#4076a9", "#df9037"],
        )
        axes[1].bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
    axes[0].set(
        xlabel="Time in diagnostic capture (s)",
        ylabel="Teacher-forced one-step normalized MSE",
        title="Each prediction starts at the true state (all transitions)",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].set(
        xticks=[i * 3 + 0.5 for i in range(3)],
        xticklabels=[c["label"] for c in cfg["cases"]],
        ylabel="Normalized state MSE",
        title="Blue: teacher-forced 1 step; orange: autoregressive 32 steps",
    )
    axes[1].tick_params(axis="x", labelsize=8)
    fig.suptitle("Different prediction protocols and sample sets; descriptive comparisons only")
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"02_one_step_and_rollout.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def report(cfg, rows, out):
    lines = [
        "# Experiment 005: World Model Prediction Analysis",
        "",
        "Frozen pretrained World Model; no training. New short diagnostic transitions are required because Experiment 004 did not log velocities/actions/context.",
        "",
        "## Protocol",
        "",
        "Teacher forcing: predict each real s[t+1] from real s[t], executed a[t] and measured xi[t]. Autoregression: initialize at real s[t], then recursively apply the SAME logged actions and xi for h steps; the policy is not rerun in predicted states.",
        "",
        "Primary autoregressive curves use shared start indices for horizons 1,4,8,16,32. Windows containing any capacity overflow in the full 32 steps are excluded for every horizon. Inclusive metrics and one-step-only overflow filters remain in metrics.csv.",
        "",
        "## Actual captures",
        "",
        "| Motion | Transitions | Stop | Overflow transitions | Max qpos difference vs Exp004 overlap |",
        "|---|---:|---|---:|---:|",
    ]
    for case in cfg["cases"]:
        meta = json.loads((out / case["motion"] / "capture.json").read_text())
        lines.append(
            f"| {case['motion']} | {meta['transitions']} | {meta['stop_reason']} | {meta['overflow_transitions']} | {meta['exp004_overlap_max_abs_qpos_difference']:.6g} |"
        )
    lines += [
        "",
        "## Prediction errors",
        "",
        "One-step rows use all overflow-free transitions; AR32 uses shared 32-step windows. Their sample counts differ.",
        "",
        "| Motion | TF1 normalized MSE | AR32 normalized MSE | AR32 root pos RMSE (m) | AR32 orientation RMSE (rad) | AR32 joint RMSE (rad) | Clean AR windows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cfg["cases"]:
        tf = next(
            r
            for r in rows
            if r["motion"] == case["motion"]
            and r["protocol"] == "teacher_forced"
            and r["subset"] == "no_overflow_transition"
        )
        ar = next(
            r
            for r in rows
            if r["motion"] == case["motion"]
            and r["protocol"] == "autoregressive"
            and r["subset"] == "no_overflow_32"
            and r["horizon"] == 32
        )

        def fmt(value):
            return "N/A" if value is None else f"{value:.6g}"

        lines.append(
            f"| {case['motion']} | {fmt(tf['normalized_state_mse'])} | {fmt(ar['normalized_state_mse'])} | {fmt(ar['root_position_rmse_m'])} | {fmt(ar['root_orientation_rmse_rad'])} | {fmt(ar['joint_position_rmse_rad'])} | {ar['samples']} |"
        )
    lines += [
        "",
        "## Metric definitions",
        "",
        "- State is the community 71-D origin-free [qpos|qvel], root quaternion wxyz, root linear velocity in world coordinates and angular velocity in body coordinates. No yaw/position realignment is performed.",
        "- Normalized state MSE is the community StateSpec.error_vector squared and averaged over 70 dimensions, then over samples; quaternion error is a rotation vector. Checkpoint scales: pos 0.05 m, rotation 0.1 rad, linear velocity 0.5 m/s, angular velocity 1 rad/s, joint pos 0.1 rad, joint velocity 2 rad/s.",
        "- Root position/orientation RMSE use vector squared norms averaged over windows, then square root. Joint RMSE averages squared errors over windows and all 29 joints. Velocity metrics are in metrics.csv. Body angular velocities are compared in the community state convention.",
        "- The WM uses body-frame features, integrates back to the same world/root/joint state representation and advances 0.02 s per step. xi contains measured friction, encoder biases and torso COM offsets.",
        "- Action is executed_raw (29 normalized PD position commands), not torque or processed joint-target radians. Noise is disabled; saved clean and executed actions are checked for equality.",
        "",
        "## Limits",
        "",
        "- Only three selected clips, one deterministic run each; this is not a population benchmark or uncertainty estimate.",
        "- At most 512 control steps (10.24 s) per case. Dance uses its early section, not the approximately 60-second failure seen in Experiment 004. Motion labels do not imply every window is difficult.",
        "- Terminated capture stops immediately and includes the true terminal state (auto_reset=False); windows do not cross resets or continue after failure.",
        "- The generalized state omits solver/contact caches. This tests prediction along logged MuJoCo transitions, not exact simulator cloning or a reinitialized state-write counterfactual.",
        "- Growth of multi-step error is a descriptive result, not proof of any single cause of policy failure. Distinguish model error, action sensitivity and numerical/contact effects.",
        "- These are real observations; nonmonotonic error curves and differences from Experiment 004 are retained.",
        "",
        "## Reproduce",
        "",
        "From repository root:",
        "",
        "```bash",
        "bash my_work/experiments/exp_005_world_model_prediction/run.sh analyze --device cuda:0",
        "```",
        "",
        "Analyze reuses saved transitions and does not create a MuJoCo environment. See the experiment README for initial capture and feasibility commands.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["check", "capture", "analyze", "all"], nargs="?", default="check"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = config()
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.mode in ["check", "all"]:
        feasibility(cfg)
    if args.mode in ["capture", "all"]:
        capture(cfg, args.device)
    if args.mode in ["analyze", "all"]:
        analyze(cfg, args.device)


if __name__ == "__main__":
    main()
