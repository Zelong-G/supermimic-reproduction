"""Paired, resumable random friction changes; original and final policy only."""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime

import numpy as np
import torch
import warp as wp
from common import HERE, OUT, ROOT, file_hash, output_path, save_json, settings, state_hash
from mjlab.envs import ManagerBasedRlEnv
from models import AdaptivePolicy, current_observation
from random_friction_protocol import DT, SWITCH_STEPS, schedule

from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX
from supermimic.eval.benchmark import env_kwargs, load_policy_from_checkpoint

FINAL = OUT / "joint_6h_from5000/ckpt_0080862.pt"


def utc():
    return datetime.now(UTC).isoformat()


def contact_check(env, feet, ground, requested):
    """Validate every active foot/plane contact, grouped by contact worldid."""
    d = env.sim.wp_data
    n = min(int(wp.to_torch(d.nacon)[0]), d.naconmax)
    geom = wp.to_torch(d.contact.geom)[:n]
    fr = wp.to_torch(d.contact.friction)[:n]
    efc = wp.to_torch(d.contact.efc_address)[:n]
    world = wp.to_torch(d.contact.worldid)[:n].long()
    hit = torch.isin(geom, feet).any(-1) & torch.isin(geom, ground).any(-1) & (efc >= 0).any(-1)
    counts = torch.zeros(len(requested), device=env.device, dtype=torch.long)
    mins = torch.full_like(requested, float("nan"))
    maxs = torch.full_like(requested, float("nan"))
    evidence = []
    if hit.any():
        w = world[hit]
        actual = fr[hit, :2]
        expected = requested[w, None].expand_as(actual)
        if not torch.allclose(actual, expected, atol=1e-6, rtol=0):
            raise AssertionError("Live contact friction differs from requested world friction")
        counts.scatter_add_(0, w, torch.ones_like(w))
        for i in w.unique().tolist():
            mask = w == i
            mins[i] = actual[mask].min()
            maxs[i] = actual[mask].max()
        # Sparse readable examples; numeric contact ranges are retained every step.
        first = hit.nonzero().flatten()[0]
        evidence = [
            {
                "world": int(world[first]),
                "geom": geom[first].cpu().tolist(),
                "friction": fr[first].cpu().tolist(),
                "efc_address": efc[first].cpu().tolist(),
            }
        ]
    return counts, mins, maxs, evidence


@torch.no_grad()
def collect(actor, cfg_checkpoint, files, variant, sequence_seed, control, limit):
    import mujoco

    torch.manual_seed(0)
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files],
        num_envs=len(files),
        episode_length_s=1e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        **env_kwargs(cfg_checkpoint),
    )
    cfg = make_env_cfg(args)
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        if abs(env.step_dt - DT) > 1e-9:
            raise AssertionError("Unexpected control timestep")
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        cmd.forced_frame[:] = lib.clip_start
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[-1]
        n = len(files)
        lengths = [c.length for c in lib.clips]
        horizon = min(max(lengths), limit) if limit else max(lengths)
        schedules = np.stack(
            [schedule(c.name, sequence_seed, horizon, control) for c in lib.clips], 1
        )
        local, _ = env.scene["robot"].find_geoms(FOOT_GEOM_REGEX)
        feet = env.scene["robot"].indexing.geom_ids[local].long()
        ground_ids = np.flatnonzero(env.sim.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        ground = torch.as_tensor(ground_ids, device=env.device)
        if len(feet) != 14 or not len(ground):
            raise AssertionError("Unexpected intervention geometry")
        model = env.sim.model.geom_friction
        ptr = env.sim.wp_model.geom_friction.ptr
        untouched = model.clone()
        expected_model = untouched.clone()
        initial = torch.cat([env.sim.data.qpos[:], env.sim.data.qvel[:]], -1).cpu().numpy()
        start_times = env.sim.data.time[:].clone()
        # Warp accumulates a float32 physics clock. Reproduce that arithmetic
        # rather than treating long-episode float32 drift as a missed switch.
        clock = start_times.cpu().numpy().copy()
        expected_clocks = []
        for _ in range(horizon):
            expected_clocks.append(clock.copy())
            for _ in range(4):
                clock += np.asarray(0.005, dtype=clock.dtype)
        active = torch.ones(n, device=env.device, dtype=torch.bool)
        last_action = torch.zeros((n, 29), device=env.device)
        records = {
            k: []
            for k in [
                "mu",
                "time",
                "position_error",
                "orientation_error",
                "reward",
                "actions",
                "action_change",
                "joint_velocity",
                "base_angular_velocity",
                "projected_gravity",
                "predicted_mu",
                "contact_count",
                "contact_mu_min",
                "contact_mu_max",
                "overflow",
                "terminated",
                "truncated",
                "active",
            ]
        }
        prefix = []
        proofs = [
            {
                "geometry": {
                    "feet": feet.cpu().tolist(),
                    "planes": ground_ids.tolist(),
                    "foot_priority": env.sim.mj_model.geom_priority[feet.cpu().numpy()].tolist(),
                    "plane_priority": env.sim.mj_model.geom_priority[ground_ids].tolist(),
                    "modified": "14 foot geoms, sliding coefficient only; planes and other axes unchanged",
                }
            }
        ]
        switch_log = []
        counts = np.zeros(n, dtype=int)
        failed = np.zeros(n, dtype=bool)
        for step in range(horizon):
            if not active.any():
                break
            requested = torch.as_tensor(schedules[step], device=env.device)
            if step % SWITCH_STEPS == 0:
                model[:, feet, 0] = requested[:, None]
                torch.cuda.synchronize()
                expected_model[:, feet, 0] = requested[:, None]
                if not torch.equal(model, expected_model):
                    raise AssertionError("Unexpected non-foot or other-axis parameter modification")
                if env.sim.wp_model.geom_friction.ptr != ptr:
                    raise AssertionError("Warp buffer was replaced")
                elapsed = env.sim.data.time[:] - start_times
                expected_clock = torch.as_tensor(expected_clocks[step], device=env.device)
                if not torch.allclose(
                    env.sim.data.time[:][active], expected_clock[active], atol=1e-3, rtol=0
                ):
                    raise AssertionError("Switch simulator time does not match schedule")
                switch_log.append(
                    {
                        "step": step,
                        "planned_time_s": step * DT,
                        "sim_time_s": elapsed.cpu().tolist(),
                        "expected_float32_sim_time_s": (expected_clock - start_times)
                        .cpu()
                        .tolist(),
                        "mu": requested.cpu().tolist(),
                        "active": active.cpu().tolist(),
                        "buffer_pointer_unchanged": True,
                    }
                )
            if not torch.allclose(
                model[:, feet, 0], requested[:, None].expand(-1, len(feet)), atol=1e-6, rtol=0
            ):
                raise AssertionError("Live model friction differs")
            pos = (cmd.robot_root_pos_w - (cmd.root_pos_w + env.scene.env_origins)).norm(dim=-1)
            q = cmd.robot_root_quat_w
            ref = cmd.root_quat_w
            ori = 2 * (q * ref).sum(-1).abs().clamp(0, 1).acos()
            current = current_observation(obs["policy"])
            if variant == "base":
                action = actor(obs["policy"], obs["goal"])
                hat = torch.full((n,), float("nan"), device=env.device)
            else:
                action, estimate, _, _ = actor.components(obs["policy"], obs["goal"])
                hat = estimate[:, 0]
            before = None
            if step < SWITCH_STEPS:
                before = (
                    torch.cat(
                        [
                            env.sim.data.qpos[:],
                            env.sim.data.qvel[:],
                            obs["policy"],
                            obs["goal"],
                            action,
                        ],
                        -1,
                    )
                    .cpu()
                    .numpy()
                )
            snapshot = {
                "mu": requested,
                "time": env.sim.data.time[:] - start_times,
                "position_error": pos,
                "orientation_error": ori,
                "actions": action,
                "action_change": (action - last_action).norm(dim=-1),
                "joint_velocity": current[:, 29:58],
                "base_angular_velocity": current[:, 58:61],
                "projected_gravity": current[:, 61:64],
                "predicted_mu": hat,
                "active": active,
            }
            # Snapshot CPU before env.step can mutate any observation/data views.
            snapshot = {k: v.cpu().numpy().copy() for k, v in snapshot.items()}
            obs, reward, term, trunc, _ = env.step(action)
            contacts, cmin, cmax, proof = contact_check(env, feet, ground, requested)
            snapshot.update(
                {
                    k: v.cpu().numpy().copy()
                    for k, v in {
                        "reward": reward,
                        "contact_count": contacts,
                        "contact_mu_min": cmin,
                        "contact_mu_max": cmax,
                        "overflow": env.sim.data.overflow[:],
                        "terminated": term,
                        "truncated": trunc,
                    }.items()
                }
            )
            env.sim.data.overflow[:] = 0
            if before is not None:
                prefix.append(np.concatenate([before, snapshot["reward"][:, None]], -1))
            if proof and (step % SWITCH_STEPS in [0, 1] or step == 2):
                proofs.append({"step": step, "examples": proof})
            for k, v in snapshot.items():
                records[k].append(v)
            alive = active.cpu().numpy()
            counts += alive
            failed |= alive & term.cpu().numpy()
            active &= ~(term | trunc)
            last_action = action.clone()
            if (step + 1) % 1000 == 0:
                print(
                    f"progress {variant} schedule={sequence_seed} control={control} step={step + 1} done={n - int(active.sum())}/{n}",
                    flush=True,
                )
        arrays = {k: np.stack(v) for k, v in records.items()}
        arrays["prefix"] = np.stack(prefix)
        arrays["schedule"] = schedules
        arrays["initial"] = initial
        rows = []
        for i, clip in enumerate(lib.clips):
            steps = int(counts[i])
            if not failed[i] and not limit and steps != clip.length:
                raise AssertionError("Completion differs from full benchmark clip length")
            ov = arrays["overflow"][:steps, i].astype(np.int64)
            contact_counts = arrays["contact_count"][:steps, i]
            if not contact_counts.any():
                raise AssertionError(f"No active foot contact verified for {clip.name}")
            rows.append(
                {
                    "name": clip.name,
                    "variant": variant,
                    "sequence_seed": sequence_seed,
                    "control": control,
                    "frames": clip.length,
                    "steps": steps,
                    "norm_return": float(arrays["reward"][:steps, i].sum()) / (0.1 * clip.length),
                    "failed": bool(failed[i]),
                    "censored": bool(not failed[i] and steps < clip.length),
                    "mean_position_error_m": float(arrays["position_error"][:steps, i].mean()),
                    "mean_orientation_error_rad": float(
                        arrays["orientation_error"][:steps, i].mean()
                    ),
                    "capacity_overflow_steps": int(np.count_nonzero(ov & ~1536)),
                    "solver_limit_steps": int(np.count_nonzero(ov & 1536)),
                    "initial_state_sha256": hashlib.sha256(initial[i].tobytes()).hexdigest(),
                    "verified_contact_steps": int(np.count_nonzero(contact_counts)),
                    "realized_switches": int((steps - 1) // SWITCH_STEPS),
                    "foot_slip_velocity": None,
                }
            )
        return arrays, rows, switch_log, proofs
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="random_friction_final_vs_community")
    p.add_argument("--clips", nargs="+")
    p.add_argument("--sequence-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--max-seconds", type=float, help="Censored pilot only; omit for full clips")
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    if len(set(a.sequence_seeds)) != len(a.sequence_seeds) or any(s < 0 for s in a.sequence_seeds):
        p.error("Sequence seeds must be distinct nonnegative integers")
    if a.max_seconds is not None and (not np.isfinite(a.max_seconds) or a.max_seconds <= 3):
        p.error("Pilot duration must exceed first switch at 3 seconds")
    torch.set_num_threads(2)
    target = output_path(OUT / "evaluation" / a.run_name)
    target.mkdir(parents=True, exist_ok=True)
    lock = (target / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = discover_npz(ROOT / "data/npz/lafan1", names=a.clips)
    if not files:
        raise ValueError("No existing LAFAN1 motion selected")
    s = settings()
    manifest = {
        "schema": "exp009/random-friction-v1",
        "clips": [str(f.relative_to(ROOT)) for f in files],
        "clip_sha256": {str(f.relative_to(ROOT)): file_hash(f) for f in files},
        "pretrained_sha256": file_hash(ROOT / s["pretrained"]),
        "final_sha256": file_hash(FINAL),
        "sequence_seeds": a.sequence_seeds,
        "environment_seed": 0,
        "dt": DT,
        "first_switch_s": 3.0,
        "interval_s": 3.0,
        "range": [0.2, 1.5],
        "max_seconds": a.max_seconds,
        "variants": ["base", "step_80862"],
        "physics": "Community solver10/line-search20/contact35/constraints250; no noise or other DR",
        "contact_verification": "Every step: live buffer, active foot/plane contacts by worldid; last physics substep",
        "prefix_gate": "Within-policy control/random state, history, goal, action, reward max absolute difference <=1e-4 before 3s",
        "clock": "Switch at control step multiples of 150; actual float32 accumulated physics clock checked, ideal and actual times logged",
        "source_sha256": {
            f: file_hash(HERE / f)
            for f in ["evaluate_random_friction.py", "random_friction_protocol.py", "models.py"]
        },
        "community_source_sha256": {
            f: file_hash(ROOT / f)
            for f in [
                "src/supermimic/envs/tracking_env_cfg.py",
                "src/supermimic/eval/benchmark.py",
                "src/supermimic/envs/mdp/commands.py",
            ]
        },
    }
    mp = target / "manifest.json"
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise RuntimeError("Inputs/protocol changed; use a new run name")
    save_json(mp, manifest)
    groups = target / "groups"
    groups.mkdir(exist_ok=True)
    status = {
        "pid": os.getpid(),
        "started_utc": utc(),
        "target_groups": 2 * (1 + len(a.sequence_seeds)),
        "episodes_per_group": len(files),
    }

    def update(**kw):
        status.update(kw)
        status.update(updated_utc=utc(), completed_groups=len(list(groups.glob("*.json"))))
        save_json(target / "status.json", status)

    update(state="prepared")
    if a.prepare_only:
        return
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("No visible CUDA GPU; check CUDA_VISIBLE_DEVICES on this machine")
        base, cfg = load_policy_from_checkpoint(ROOT / s["pretrained"], "cuda:0")
        frozen = state_hash(base)
        ck = torch.load(FINAL, map_location="cpu", weights_only=False)
        if (
            ck["pretrained_sha256"] != manifest["pretrained_sha256"]
            or ck["frozen_hashes"]["policy"] != frozen
        ):
            raise AssertionError("Final checkpoint does not match original policy")
        final = AdaptivePolicy(base, ck["settings"]["alpha"]).to("cuda:0").eval()
        final.load_learned(ck["learned"])
        del ck
        for variant, actor in [("base", base), ("step_80862", final)]:
            for seed in [None, *a.sequence_seeds]:
                key = f"{variant}_" + ("control" if seed is None else f"random_seed{seed}")
                path = groups / f"{key}.json"
                if path.exists():
                    continue
                update(state="evaluating", current_group=key)
                arrays, rows, switches, proofs = collect(
                    actor,
                    cfg,
                    files,
                    variant,
                    0 if seed is None else seed,
                    seed is None,
                    round(a.max_seconds / DT) if a.max_seconds else None,
                )
                npz = groups / f"{key}.npz"
                tmp = groups / f"{key}.npz.tmp"
                with tmp.open("wb") as f:
                    np.savez_compressed(f, **arrays)
                tmp.replace(npz)
                save_json(
                    path,
                    {
                        "group": key,
                        "episodes": rows,
                        "switches": switches,
                        "contact_proofs": proofs,
                        "trace_sha256": file_hash(npz),
                        "finished_utc": utc(),
                    },
                )
                del arrays
                gc.collect()
                torch.cuda.empty_cache()
                update(state="group_saved")
        if state_hash(base) != frozen or file_hash(FINAL) != manifest["final_sha256"]:
            raise AssertionError("Frozen weights changed")
        update(state="reporting")
        subprocess.run(
            [sys.executable, str(HERE / "random_friction_report.py"), "--eval-dir", str(target)],
            check=True,
            cwd=ROOT,
        )
        update(state="complete", finished_utc=utc())
    except BaseException as exc:
        update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
