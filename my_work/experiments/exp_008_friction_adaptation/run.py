"""Frozen-policy friction experiment, local wrapper around the community environment."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "my_work/outputs/exp_008_friction_adaptation"
sys.path.insert(0, str(ROOT / "src"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def prepare():
    import numpy as np

    with (ROOT / "my_work/outputs/exp_003_failure_analysis/segment_mapping.csv").open() as source:
        rows = list(csv.DictReader(source))
    turns = []
    for original in rows:
        if original["category"] != "walk":
            continue
        r = dict(original)
        with np.load(ROOT / "data/npz/lafan1_seg10" / f"{r['name']}.npz") as d:
            q = d["body_quat_w"][:, list(d["body_names"]).index("pelvis")].astype(float)
        w, x, y, z = q.T
        yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        r["yaw_range_deg"] = float(np.ptp(yaw) * 180 / np.pi)
        if r["failed"] == "False" and r["overflow_steps"] == "0":
            turns.append(r)
    turns.sort(key=lambda r: r["yaw_range_deg"], reverse=True)
    save(OUT / "turn_candidates.json", turns)
    cases = []
    # Different source trajectories for each of the three examples in a stratum.
    choices = {
        "Walking": ["walk1_subject1__seg000", "walk1_subject2__seg000", "walk1_subject5__seg000"],
        "Dancing": [
            "dance1_subject1__seg000",
            "dance1_subject3__seg000",
            "dance2_subject4__seg005",
        ],
        "Fall-and-Get-Up": [
            "fallAndGetUp1_subject1__seg000",
            "fallAndGetUp2_subject2__seg000",
            "fallAndGetUp3_subject1__seg000",
        ],
    }
    used = set()
    choices["Turning"] = []
    for r in turns:
        if r["source_motion"] in used or r["yaw_range_deg"] < 45:
            continue
        used.add(r["source_motion"])
        choices["Turning"].append(r["name"])
        if len(used) == 3:
            break
    if len(used) != 3:
        raise ValueError("Need three verified turning trajectories")
    lookup = {r["name"]: r for r in rows}
    for category, names in choices.items():
        for name in names:
            r = lookup[name]
            p = ROOT / "data/npz/lafan1_seg10" / f"{name}.npz"
            d = np.load(p)
            q = d["body_quat_w"][:, list(d["body_names"]).index("pelvis")]
            w, x, y, z = q.T
            yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            cases.append(
                {
                    "name": name,
                    "category": category,
                    "source": r["source_motion"],
                    "path": str(p.relative_to(ROOT)),
                    "frames": int(r["frames"]),
                    "yaw_range_deg": float(np.ptp(yaw) * 180 / np.pi),
                    "historical_score": float(r["norm_return"]),
                    "historical_failed": r["failed"] == "True",
                    "historical_overflow": int(r["overflow_steps"]),
                    "sha256": sha(p),
                }
            )
    config = {
        "checkpoint": "weights/realmimic_g1_86k.pt",
        "seed": 0,
        "frictions": [1.0, 0.7, 0.5, 0.3, 0.15],
        "cases": cases,
        "switch_step": 150,
        "protocol": "Existing independent 10 s LAFAN1 segments; full-segment denominator; stop on first termination; no noise or other DR; one deterministic seed, no pseudo-replicated seeds.",
        "recovery": "First post-switch start of 50 consecutive steps with paired excess root position error <=0.05 m and excess orientation error <=0.10 rad; meaningful only after a prior violation, otherwise no detected disruption. Right-censored at failure or segment end.",
    }
    save(HERE / "config.json", config)
    print(json.dumps(config, indent=2))


def capture(stage, device):
    import mujoco
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from probe import contact_rows

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX
    from supermimic.eval.benchmark import env_kwargs, load_policy_from_checkpoint

    config = json.loads((HERE / "config.json").read_text())
    torch.set_num_threads(2)
    configure_torch_backends()
    policy, ck = load_policy_from_checkpoint(ROOT / config["checkpoint"], device)
    cases = (
        config["cases"]
        if stage in ["fixed", "repair"]
        else [c for c in config["cases"] if c["category"] == "Walking"]
    )
    if stage == "repair":
        affected = set()
        for p in (OUT / "fixed").glob("*__mu*.json"):
            r = json.loads(p.read_text())
            if r["capacity_overflow_steps"]:
                affected.add(r["name"])
        cases = [c for c in cases if c["name"] in affected]
        if not cases:
            print("No capacity-affected cases to repair")
            return
    if stage.startswith("pilot"):
        cases = [config["cases"][0]]
    files = [str(ROOT / c["path"]) for c in (config["cases"] if stage == "repair" else cases)]
    cfg = make_env_cfg(
        SuperMimicEnvCfgArgs(
            npz_files=files,
            num_envs=1,
            episode_length_s=1e5,
            action_noise_std=0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            sampling_mode="start",
            record_transitions=False,
            seed=config["seed"],
            **env_kwargs(ck),
        )
    )
    cfg.auto_reset = False
    if stage in ["pilot_high", "repair", "switch"]:
        cfg.sim.nconmax = 140
        cfg.sim.njmax = 1000
    if stage in ["pilot_solver", "fixed", "switch", "repair"]:
        cfg.sim.mujoco.ls_iterations = 80
    if stage in ["pilot_precision", "fixed", "switch", "repair"]:
        cfg.sim.mujoco.iterations = 100
        cfg.sim.mujoco.ls_iterations = 80
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        io = StateIO(env)
        local, _names = env.scene["robot"].find_geoms(FOOT_GEOM_REGEX)
        feet = env.scene["robot"].indexing.geom_ids[local].long().tolist()
        m = env.sim.mj_model
        ground = np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist()
        model_info = [
            {
                "id": i,
                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i),
                "priority": int(m.geom_priority[i]),
                "condim": int(m.geom_condim[i]),
                "friction": m.geom_friction[i].tolist(),
            }
            for i in ground + feet
        ]
        mode_dir = OUT / ("fixed" if stage == "repair" else stage)
        save(
            mode_dir / ("repair_metadata.json" if stage == "repair" else "metadata.json"),
            {
                "checkpoint_sha256": sha(ROOT / config["checkpoint"]),
                "checkpoint_policy": ck["policy"],
                "checkpoint_env": ck["env"],
                "env_kwargs": env_kwargs(ck),
                "geoms": model_info,
                "npair": int(m.npair),
                "physics_dt": env.physics_dt,
                "control_dt": env.step_dt,
                "nconmax": cfg.sim.nconmax,
                "njmax": cfg.sim.njmax,
                "iterations": cfg.sim.mujoco.iterations,
                "ls_iterations": cfg.sim.mujoco.ls_iterations,
                "auto_reset": False,
                "history_order": "term-major, oldest to newest within each term",
                "intervention": "in-place write to existing expanded Warp geom_friction buffer, tangential axis only, 14 foot geoms; ground unchanged",
                "selected_cases": [c["name"] for c in cases],
                "config": config,
            },
        )
        if stage == "repair":
            archive = OUT / "capacity_affected_initial"
            archive.mkdir(exist_ok=True)
            for c in cases:
                for mu in config["frictions"]:
                    for ext in ["npz", "json"]:
                        p = mode_dir / f"{c['name']}__mu{mu:g}.{ext}"
                        if (archive / p.name).exists():
                            raise FileExistsError(archive / p.name)
                        p.rename(archive / p.name)
        conditions = (
            [1.0, 1.0, 0.3, 0.15]
            if stage == "switch"
            else [1.0, 0.3, 0.15]
            if stage == "pilot_precision"
            else [1.0, 0.3]
            if stage.startswith("pilot")
            else config["frictions"]
        )
        for case in cases:
            ci = next(i for i, c in enumerate(lib.clips) if c.name == case["name"])
            for condition_index, mu in enumerate(conditions):
                suffix = "_repeat" if stage == "switch" and condition_index == 1 else ""
                path = mode_dir / f"{case['name']}__mu{mu:g}{suffix}"
                if Path(str(path) + ".npz").exists():
                    raise FileExistsError(path)
                torch.manual_seed(config["seed"])
                cmd.forced_frame[:] = lib.clip_start[ci]
                obs, _ = env.reset()
                env.sim.model.geom_friction[:, feet, 0] = 1.0 if stage == "switch" else mu
                torch.cuda.synchronize()
                records = {
                    k: []
                    for k in [
                        "states",
                        "reference",
                        "history",
                        "goal",
                        "actions",
                        "reward",
                        "position_error",
                        "orientation_error",
                        "mu",
                        "sim_time",
                        "contact_count",
                        "contact_mu_min",
                        "contact_mu_max",
                        "active_foot_contacts",
                        "overflow",
                        "overflow_mask",
                        "terminated",
                        "truncated",
                    ]
                }
                evidence = []
                initial = io.read()[0].cpu().numpy()
                initial_time = float(env.sim.data.time[0])
                friction_pointer = env.sim.wp_model.geom_friction.ptr
                first_active_saved = False
                for k in range(case["frames"]):
                    if stage == "switch" and k == config["switch_step"]:
                        env.sim.model.geom_friction[:, feet, 0] = mu
                        torch.cuda.synchronize()
                    actual = float(env.sim.model.geom_friction[0, feet[0], 0])
                    assert env.sim.wp_model.geom_friction.ptr == friction_pointer
                    assert torch.allclose(
                        env.sim.model.geom_friction[0, feet, 0],
                        torch.full((len(feet),), actual, device=device),
                    )
                    sim_time = float(env.sim.data.time[0]) - initial_time
                    assert abs(sim_time - k * env.step_dt) < 1e-3
                    state = io.read()[0].cpu().numpy()
                    ref = (
                        lib.ref_qpos_qvel(lib.clip_start[ci] + torch.tensor([k], device=device))[0][
                            0
                        ]
                        .cpu()
                        .numpy()
                    )
                    q = state[3:7].astype(float)
                    r = ref[3:7].astype(float)
                    angle = 2 * np.arccos(
                        np.clip(abs(q @ r) / (np.linalg.norm(q) * np.linalg.norm(r)), 0, 1)
                    )
                    records["states"].append(state)
                    records["reference"].append(ref)
                    records["position_error"].append(np.linalg.norm(state[:3] - ref[:3]))
                    records["orientation_error"].append(angle)
                    records["history"].append(obs["policy"][0].cpu().numpy())
                    records["goal"].append(obs["goal"][0].cpu().numpy())
                    with torch.inference_mode():
                        action = policy(obs["policy"], obs["goal"])
                    records["actions"].append(action[0].cpu().numpy())
                    obs, reward, term, trunc, _ = env.step(action)
                    contacts = contact_rows(env, feet, ground)
                    active = [c for c in contacts if any(a >= 0 for a in c["efc_address"])]
                    mus = [c["friction"][0] for c in contacts]
                    if active and not all(
                        np.isclose(c["friction"][0], actual, atol=1e-6) for c in active
                    ):
                        save(
                            mode_dir / "FAILED_contact_validation.json",
                            {"step": k, "mu": actual, "contacts": contacts},
                        )
                        raise RuntimeError(
                            "Requested foot friction is not effective contact friction"
                        )
                    if k in [0, 1, 149, 150, 151, 152, 160] or (active and not first_active_saved):
                        evidence.append(
                            {
                                "step": k,
                                "time_s": k * env.step_dt,
                                "actual_sim_time_s": sim_time,
                                "requested_mu": actual,
                                "contacts": contacts,
                            }
                        )
                        if active:
                            first_active_saved = True
                    for key, val in {
                        "reward": float(reward[0]),
                        "mu": actual,
                        "sim_time": sim_time,
                        "contact_count": len(contacts),
                        "contact_mu_min": min(mus) if mus else -1.0,
                        "contact_mu_max": max(mus) if mus else -1.0,
                        "active_foot_contacts": len(active),
                        "overflow": bool((env.sim.data.overflow[0] != 0).item()),
                        "overflow_mask": int(env.sim.data.overflow[0]),
                        "terminated": bool(term[0]),
                        "truncated": bool(trunc[0]),
                    }.items():
                        records[key].append(val)
                    env.sim.data.overflow[:] = 0
                    if bool(term[0]) or bool(trunc[0]):
                        break
                arrays = {key: np.asarray(val) for key, val in records.items()}
                np.savez_compressed(Path(str(path) + ".npz"), **arrays)
                steps = len(arrays["reward"])
                failed = bool(arrays["terminated"][-1])
                row = {
                    "stage": stage,
                    "name": case["name"],
                    "category": case["category"],
                    "source": case["source"],
                    "seed": config["seed"],
                    "mu": mu,
                    "frames": case["frames"],
                    "steps": steps,
                    "norm_return": float(arrays["reward"].sum() / (0.1 * case["frames"])),
                    "failed": failed,
                    "time_to_failure_s": steps * env.step_dt if failed else None,
                    "observed_duration_s": steps * env.step_dt,
                    "mean_root_position_error_m": float(arrays["position_error"].mean()),
                    "mean_root_orientation_error_rad": float(arrays["orientation_error"].mean()),
                    "overflow_steps": int(arrays["overflow"].sum()),
                    "capacity_overflow_steps": int(np.sum((arrays["overflow_mask"] & 511) != 0)),
                    "solver_limit_steps": int(np.sum((arrays["overflow_mask"] & 1536) != 0)),
                    "active_contact_samples": int(np.sum(arrays["active_foot_contacts"] > 0)),
                    "initial_state": initial.tolist(),
                    "contact_evidence": evidence,
                }
                save(Path(str(path) + ".json"), row)
                print(
                    stage,
                    case["name"],
                    mu,
                    steps,
                    round(row["norm_return"], 4),
                    "FAIL" if failed else "PASS",
                    "overflow",
                    row["overflow_steps"],
                    flush=True,
                )
    finally:
        env.close()
        del env
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "prepare",
            "pilot",
            "pilot_high",
            "pilot_solver",
            "pilot_precision",
            "fixed",
            "repair",
            "switch",
            "analyze",
        ],
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.mode == "prepare":
        prepare()
    elif args.mode == "analyze":
        from analyze import analyze

        analyze()
    else:
        capture(args.mode, args.device)


if __name__ == "__main__":
    main()
