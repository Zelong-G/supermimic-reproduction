"""LAFAN1-only fixed-friction comparison, standard full-clip denominator and terminations."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
from pathlib import Path

import numpy as np
import torch
import warp as wp
from common import OUT, ROOT, file_hash, output_path, save_json, settings
from mjlab.envs import ManagerBasedRlEnv
from models import AdaptivePolicy

from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX
from supermimic.eval.benchmark import (
    ClipResult,
    env_kwargs,
    load_policy_from_checkpoint,
    save_results,
)


def verify_contacts(env, feet, ground, mu):
    d = env.sim.wp_data
    n = min(int(wp.to_torch(d.nacon)[0]), d.naconmax)
    geom = wp.to_torch(d.contact.geom)[:n]
    fr = wp.to_torch(d.contact.friction)[:n]
    efc = wp.to_torch(d.contact.efc_address)[:n]
    ids = torch.as_tensor(feet, device=env.device)
    gids = torch.as_tensor(ground, device=env.device)
    hit = torch.isin(geom, ids).any(-1) & torch.isin(geom, gids).any(-1) & (efc >= 0).any(-1)
    if hit.any():
        actual = fr[hit, :2]
        if not torch.allclose(actual, torch.full_like(actual, mu), atol=1e-6, rtol=0):
            raise AssertionError("Requested friction not effective in active contacts")
        return {
            "active_contacts": int(hit.sum()),
            "friction": fr[hit][0].cpu().tolist(),
            "geoms": geom[hit][0].cpu().tolist(),
            "efc_address": efc[hit][0].cpu().tolist(),
        }
    return None


@torch.no_grad()
def chunk(policy, files, ck, mu, variant, seed):
    import mujoco

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
        seed=seed,
        **env_kwargs(ck),
    )
    cfg = make_env_cfg(args)
    cfg.events["foot_friction"].params["ranges"] = (mu, mu)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n = len(files)
        cmd.forced_frame[:] = lib.clip_start
        obs, _ = env.reset()
        # Community queue behavior for one world per clip: exhausted worlds
        # reset onto the final clip while remaining worlds finish their first episode.
        cmd.forced_frame[:] = lib.clip_start[-1]
        local, _ = env.scene["robot"].find_geoms(FOOT_GEOM_REGEX)
        feet = env.scene["robot"].indexing.geom_ids[local].long().tolist()
        ground = np.flatnonzero(env.sim.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist()
        assert torch.allclose(
            env.sim.model.geom_friction[:, feet, 0],
            torch.full((n, len(feet)), mu, device=env.device),
        )
        initial = torch.cat([env.sim.data.qpos[:], env.sim.data.qvel[:]], -1).cpu().numpy()
        active = torch.ones(n, device=env.device, dtype=torch.bool)
        count = torch.zeros(n, device=env.device, dtype=torch.long)
        first_ov = torch.full((n,), -1, device=env.device, dtype=torch.long)
        bits_seen = torch.zeros(n, device=env.device, dtype=torch.long)
        sums = {
            k: torch.zeros(n, device=env.device)
            for k in [
                "reward",
                "body_position_error",
                "joint_error",
                "mu_error",
                "overflow",
                "capacity_overflow",
                "solver_limit",
            ]
        }
        contact_proofs = []
        predictions = []
        results = []
        details = []
        step = 0
        max_h = torch.zeros(n, device=env.device)
        max_t = torch.zeros(n, device=env.device)
        while active.any():
            mask = active.float()
            # Pre-action diagnostics, same target phase and world frame as community command.
            bp = (cmd.robot_body_pos_w - cmd.body_pos_w).norm(dim=-1).mean(-1)
            je = (cmd.robot_joint_pos - cmd.joint_pos).square().mean(-1).sqrt()
            if variant == "base":
                action = policy(obs["policy"], obs["goal"])
                hat = None
            else:
                action, hat, _, _ = policy.components(obs["policy"], obs["goal"])
                if variant == "residual_disabled":
                    assert torch.equal(action, policy.base(obs["policy"], obs["goal"]))
                sums["mu_error"] += (hat[:, 0] - mu).abs() * mask
                if step % 25 == 0:
                    predictions.extend(
                        {
                            "variant": variant,
                            "mu": mu,
                            "name": lib.clips[i].name,
                            "step": step,
                            "predicted_mu": float(hat[i, 0]),
                        }
                        for i in active.nonzero().flatten().tolist()
                    )
            obs, reward, term, tout, _ = env.step(action)
            max_h = torch.maximum(max_h, cmd.root_height_error * mask)
            max_t = torch.maximum(max_t, cmd.root_tilt_error * mask)
            sums["reward"] += reward * mask
            sums["body_position_error"] += bp * mask
            sums["joint_error"] += je * mask
            ov = env.sim.data.overflow[:].clone()
            sums["overflow"] += (ov != 0) * mask
            sums["capacity_overflow"] += ((ov & ~1536) != 0) * mask
            sums["solver_limit"] += ((ov & 1536) != 0) * mask
            first_ov = torch.where(active & (ov != 0) & (first_ov < 0), count + 1, first_ov)
            bits_seen |= ov.long() * active.long()
            env.sim.data.overflow[:] = 0
            count += active.long()
            # Inspect actual solver contacts every step, not only the startup setting.
            proof = verify_contacts(env, feet, ground, mu)
            if proof and (not contact_proofs or step % 500 == 0):
                contact_proofs.append({"step": step, **proof})
            done = active & (term | tout)
            for i in done.nonzero().flatten().tolist():
                length = lib.clips[i].length
                steps = int(count[i])
                failed = bool(term[i])
                if not failed and steps != length:
                    raise AssertionError("Completion differs from benchmark clip length")
                result = ClipResult(
                    name=lib.clips[i].name,
                    frames=length,
                    steps=steps,
                    norm_return=float(sums["reward"][i]) / (0.1 * max(length, steps)),
                    norm_return_survived=float(sums["reward"][i]) / (0.1 * steps),
                    failed=failed,
                    fail_step=steps if failed else -1,
                    overflow_steps=int(sums["overflow"][i]),
                    first_overflow_step=int(first_ov[i]),
                    max_height_error=float(max_h[i]),
                    max_tilt_error=float(max_t[i]),
                )
                results.append(result)
                details.append(
                    {
                        "variant": variant,
                        "mu": mu,
                        "ood_stress_test": mu < 0.3,
                        "name": result.name,
                        "seed": seed,
                        "frames": length,
                        "steps": steps,
                        "norm_return": result.norm_return,
                        "failed": failed,
                        "friction_mae": None
                        if variant == "base"
                        else float(sums["mu_error"][i] / steps),
                        "mean_body_position_error_m": float(sums["body_position_error"][i] / steps),
                        "mean_joint_rmse_rad": float(sums["joint_error"][i] / steps),
                        "overflow_steps": int(sums["overflow"][i]),
                        "capacity_overflow_steps": int(sums["capacity_overflow"][i]),
                        "solver_limit_steps": int(sums["solver_limit"][i]),
                        "overflow_mask_or": int(bits_seen[i]),
                        "foot_slip_velocity": None,
                        "foot_slip_note": "Not available: no validated contact-point velocity estimator",
                        "initial_state_sha256": hashlib.sha256(initial[i].tobytes()).hexdigest(),
                    }
                )
            active &= ~done
            step += 1
            if step % 1000 == 0:
                print(
                    f"progress {variant} mu={mu:g} step={step} completed={len(results)}/{n}",
                    flush=True,
                )
        if not contact_proofs:
            raise AssertionError("No effective foot-ground contacts verified")
        return results, details, predictions, contact_proofs
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--clips", nargs="+")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ood", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    target = output_path(OUT / "evaluation" / args.run_name)
    if target.exists():
        raise FileExistsError(target)
    learned = torch.load(output_path(args.checkpoint), map_location="cpu", weights_only=False)
    if args.checkpoint.name.startswith("smoke"):
        raise ValueError("Validation-only checkpoint is not a fine-tuned evaluation model")
    if file_hash(ROOT / learned["pretrained"]) != learned["pretrained_sha256"]:
        raise ValueError("Original checkpoint changed")
    base, ck = load_policy_from_checkpoint(ROOT / learned["pretrained"], "cuda:0")
    adaptive = AdaptivePolicy(base, learned["settings"]["alpha"]).to("cuda:0")
    adaptive.load_learned(learned["learned"])
    adaptive.eval()
    files = discover_npz(ROOT / settings()["eval_data"], names=args.clips)
    if not files:
        raise FileNotFoundError("No existing LAFAN1 clips selected")
    for f in files:
        if not f.resolve().is_relative_to((ROOT / "data/npz/lafan1").resolve()):
            raise ValueError("Evaluation must use LAFAN1 only")
    target.mkdir(parents=True)
    rows = []
    preds = []
    initial = {}
    frictions = settings()["eval_frictions"] + ([0.2] if args.ood else [])
    for mu in frictions:
        for variant in ["base", "adapted", "residual_disabled"]:
            adaptive.residual_enabled = variant != "residual_disabled"
            actor = base if variant == "base" else adaptive
            results = []
            proofs = []
            for i in range(0, len(files), settings()["eval_num_envs"]):
                torch.manual_seed(args.seed)
                r, d, pred, proof = chunk(
                    actor, files[i : i + settings()["eval_num_envs"]], ck, mu, variant, args.seed
                )
                for row in d:
                    key = (mu, row["name"])
                    if key in initial:
                        assert initial[key] == row["initial_state_sha256"], (
                            "Unpaired evaluation initial state"
                        )
                    else:
                        initial[key] = row["initial_state_sha256"]
                results += r
                rows += d
                preds += pred
                proofs += proof
                gc.collect()
                torch.cuda.empty_cache()
            meta = {
                "variant": variant,
                "mu": mu,
                "ood": mu < 0.3,
                "checkpoint": str(args.checkpoint),
                "iteration": learned["iteration"],
                "contact_proofs": proofs,
                "protocol": "Community full-clip denominator, no noise or unrelated DR, start frame, original physics (10 solver /20 line search /35 contacts /250 constraints). Capacity flags retained, not treated as valid robustness evidence.",
            }
            save_results(results, target / f"{variant}_mu{mu:g}.json", meta)
            print(
                variant,
                mu,
                "clips",
                len(results),
                "mean_return",
                sum(r.norm_return for r in results) / len(results),
                flush=True,
            )
            save_json(target / "episodes.json", rows)
    with (target / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    save_json(target / "friction_predictions.json", preds)
    save_json(
        target / "protocol.json",
        {
            "clips": [str(p.relative_to(ROOT)) for p in files],
            "frictions": frictions,
            "seed": args.seed,
            "checkpoint": str(args.checkpoint),
            "iteration": learned["iteration"],
            "pretrained_sha256": learned["pretrained_sha256"],
            "residual_disabled_action_equivalence": "checked exactly on every identical observation",
            "initial_states_paired": True,
            "body_errors": "Mean world-space tracked-body position error; joint RMSE; pre-action, conditional on surviving steps.",
            "physics": "Unchanged standard benchmark; capacity-affected episodes are flagged; numerical control runs needed before interpreting them.",
        },
    )


if __name__ == "__main__":
    main()
