"""Small, real Warp contact probe. No policy training or source patching."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import json

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv

from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX


def contact_rows(env, feet, ground):
    d = env.sim.wp_data
    n = int(wp.to_torch(d.nacon)[0])
    c = d.contact
    g = wp.to_torch(c.geom)[:n].cpu().numpy()
    f = wp.to_torch(c.friction)[:n].cpu().numpy()
    e = wp.to_torch(c.efc_address)[:n].cpu().numpy()
    dist = wp.to_torch(c.dist)[:n].cpu().numpy()
    rows = []
    for i, pair in enumerate(g):
        if any(int(x) in ground for x in pair) and any(int(x) in feet for x in pair):
            rows.append(
                {
                    "geoms": pair.tolist(),
                    "friction": f[i].tolist(),
                    "efc_address": e[i].tolist(),
                    "dist": float(dist[i]),
                }
            )
    return rows


def main():
    torch.set_num_threads(2)
    cfg = make_env_cfg(
        SuperMimicEnvCfgArgs(
            npz_files=[str(ROOT / "data/npz/lafan1/walk1_subject1.npz")],
            num_envs=1,
            domain_randomization=False,
            obs_noise=False,
            action_noise_std=0,
            reset_noise=False,
            record_transitions=False,
            sampling_mode="start",
        )
    )
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        m = env.sim.mj_model
        local, _names = env.scene["robot"].find_geoms(FOOT_GEOM_REGEX)
        feet = env.scene["robot"].indexing.geom_ids[local].long().tolist()
        ground = np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist()
        info = [
            {
                "id": i,
                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i),
                "friction": m.geom_friction[i].tolist(),
                "priority": int(m.geom_priority[i]),
                "condim": int(m.geom_condim[i]),
            }
            for i in ground + feet
        ]
        print("GEOMS", json.dumps(info), flush=True)
        print("MODEL", env.sim.model.geom_friction.shape, "npair", m.npair, flush=True)
        _obs, _ = env.reset()
        samples = []
        for mu in [1.0, 0.3, 0.15]:
            env.sim.model.geom_friction[:, feet, 0] = mu
            for k in range(3):
                env.step(torch.zeros((1, 29), device="cuda:0"))
                samples.append({"mu": mu, "step": k, "contacts": contact_rows(env, feet, ground)})
        report = {"geoms": info, "npair": int(m.npair), "samples": samples}
        (ROOT / "my_work/outputs/exp_008_friction_adaptation/contact_probe.json").write_text(
            json.dumps(report, indent=2)
        )
        print(json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
