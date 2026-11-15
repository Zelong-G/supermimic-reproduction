"""Resumable, GPU-idle queued comparison of four fixed checkpoints on full LAFAN1.

One shared base and one disabled-residual numerical control; no new training.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime

import torch
from common import HERE, OUT, ROOT, file_hash, output_path, save_json, settings, state_hash
from evaluate import chunk
from models import AdaptivePolicy

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint, summarize

CHECKPOINTS = {
    "step_5000": "joint_seed0/ckpt_0005000.pt",
    "step_10000": "joint_6h_from5000/ckpt_0010000.pt",
    "step_20000": "joint_6h_from5000/ckpt_0020000.pt",
    "step_80862": "joint_6h_from5000/ckpt_0080862.pt",
}
ORDER = ["base", "step_80862", "step_5000", "step_10000", "step_20000", "residual_disabled"]


def utc():
    return datetime.now(UTC).isoformat()


def gpu_others():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    result = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        pid = int(fields[0].strip())
        if pid != os.getpid():
            result.append({"pid": pid, "memory_mib": fields[1].strip()})
    return result


def group_key(variant, mu):
    return f"{variant}_mu{mu:g}"


def manifest(files):
    s = settings()
    return {
        "schema": "exp009/checkpoint-sweep-v1",
        "checkpoints": {
            k: {"path": v, "sha256": file_hash(OUT / v)} for k, v in CHECKPOINTS.items()
        },
        "pretrained": s["pretrained"],
        "pretrained_sha256": file_hash(ROOT / s["pretrained"]),
        "clips": [str(p.relative_to(ROOT)) for p in files],
        "clip_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in files},
        "frictions": s["eval_frictions"],
        "variants": ORDER,
        "seed": 0,
        "eval_chunk_size": s["eval_num_envs"],
        "physics": "Original community 0.005s x4, nconmax35, njmax250, solver10, line-search20, start frames, no noise/other DR; actual foot contacts verified",
        "shared_controls": "Original base and disabled-residual 5k wrapper are shared across all checkpoints; disabled action equality checked at each identical observation.",
        "source_sha256": {
            p: file_hash(ROOT / p)
            for p in [
                "src/supermimic/envs/tracking_env_cfg.py",
                "src/supermimic/eval/benchmark.py",
                str((HERE / "evaluate.py").relative_to(ROOT)),
                str((HERE / "models.py").relative_to(ROOT)),
            ]
        },
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="sweep_5k_10k_20k_80862")
    p.add_argument("--final-only", action="store_true", help="Compare only original and final 80,862 checkpoint")
    p.add_argument("--frictions", nargs="+", type=float, help="Explicit positive friction coefficients")
    p.add_argument("--reuse-from", help="Reuse compatible completed groups from another run name")
    p.add_argument("--prepare-only", action="store_true", help="Validate inputs and reuse groups without loading CUDA")
    p.add_argument(
        "--share-gpu",
        action="store_true",
        help="Run immediately alongside other GPU processes when explicitly authorized",
    )
    a = p.parse_args()
    order = ["base", "step_80862"] if a.final_only else ORDER
    if a.frictions and (len(set(a.frictions)) != len(a.frictions) or any(not math.isfinite(x) or x <= 0 for x in a.frictions)):
        p.error("Friction values must be finite, positive and unique")
    torch.set_num_threads(2)
    target = output_path(OUT / "evaluation" / a.run_name)
    target.mkdir(parents=True, exist_ok=True)
    lock = (target / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = discover_npz(ROOT / "data/npz/lafan1")
    m = manifest(files)
    if a.final_only:
        m["variants"] = order
        m["checkpoints"] = {"step_80862": m["checkpoints"]["step_80862"]}
        m["shared_controls"] = "Original and final checkpoint only; no disabled-residual numerical repeat in this sweep."
    if a.frictions:
        m["frictions"] = a.frictions
    mp = target / "manifest.json"
    if mp.exists() and json.loads(mp.read_text()) != m:
        raise RuntimeError("Sweep input/protocol changed; choose a fresh run name")
    save_json(mp, m)
    groups = target / "groups"
    groups.mkdir(exist_ok=True)
    if a.reuse_from:
        source = output_path(OUT / "evaluation" / a.reuse_from)
        old = json.loads((source / "manifest.json").read_text())
        flexible = {"variants", "frictions", "checkpoints", "shared_controls"}
        if {k: v for k, v in m.items() if k not in flexible} != {k: v for k, v in old.items() if k not in flexible}:
            raise RuntimeError("Reuse source has incompatible data, physics or source hashes")
        if any(old["checkpoints"].get(k) != v for k, v in m["checkpoints"].items()):
            raise RuntimeError("Reuse source checkpoint differs")
        for mu in m["frictions"]:
            for variant in order:
                key = group_key(variant, mu)
                src = source / "groups" / f"{key}.json"
                dst = groups / src.name
                if src.exists() and not dst.exists():
                    payload = json.loads(src.read_text())
                    if payload["variant"] != variant or payload["mu"] != mu or len(payload["episodes"]) != len(files):
                        raise RuntimeError(f"Invalid reuse group: {src}")
                    payload["reused_from"] = str(src.relative_to(ROOT))
                    payload["reused_source_sha256"] = file_hash(src)
                    save_json(dst, payload)
    started = utc()
    status = {
        "pid": os.getpid(),
        "started_utc": started,
        "target_groups": len(order) * len(m["frictions"]),
        "episodes_per_group": len(files),
        "state": "prepared",
        "gpu_sharing_authorized": a.share_gpu,
    }

    def update(**kw):
        status.update(kw)
        status["updated_utc"] = utc()
        status["completed_groups"] = len(list(groups.glob("*.json")))
        save_json(target / "status.json", status)

    def wait_gpu():
        if a.share_gpu:
            return
        while True:
            other = gpu_others()
            if not other:
                return
            update(state="waiting_for_gpu_idle", other_gpu_processes=other)
            time.sleep(15)

    if a.prepare_only:
        update(state="prepared")
        print(json.dumps(status), flush=True)
        return

    try:
        update(state="waiting_for_gpu_idle")
        wait_gpu()
        base, cfg = load_policy_from_checkpoint(ROOT / m["pretrained"], "cuda:0")
        adapter = AdaptivePolicy(base, 0.1).to("cuda:0").eval()
        frozen = state_hash(base)
        for mu in m["frictions"]:
            for variant in order:
                key = group_key(variant, mu)
                path = groups / f"{key}.json"
                if path.exists():
                    continue
                wait_gpu()
                update(state="evaluating", current_group=key, other_gpu_processes=gpu_others())
                if variant == "base":
                    actor = base
                else:
                    label = "step_5000" if variant == "residual_disabled" else variant
                    ck = torch.load(
                        OUT / CHECKPOINTS[label], map_location="cpu", weights_only=False
                    )
                    if (
                        ck["frozen_hashes"]["policy"] != frozen
                        or ck["pretrained_sha256"] != m["pretrained_sha256"]
                    ):
                        raise AssertionError("Frozen pretrained state differs across checkpoints")
                    adapter.load_learned(ck["learned"])
                    adapter.alpha = ck["settings"]["alpha"]
                    adapter.residual_enabled = variant != "residual_disabled"
                    actor = adapter
                    del ck
                results = []
                rows = []
                predictions = []
                proofs = []
                tick = time.perf_counter()
                for start in range(0, len(files), m["eval_chunk_size"]):
                    torch.manual_seed(m["seed"])
                    r, d, pred, proof = chunk(
                        actor,
                        files[start : start + m["eval_chunk_size"]],
                        cfg,
                        mu,
                        variant,
                        m["seed"],
                    )
                    results += r
                    rows += d
                    predictions += pred
                    proofs += proof
                if variant != "base":
                    baseline = json.loads((groups / f"{group_key('base', mu)}.json").read_text())
                    hashes = {r["name"]: r["initial_state_sha256"] for r in baseline["episodes"]}
                    assert all(hashes[r["name"]] == r["initial_state_sha256"] for r in rows), (
                        "Initial states are unpaired"
                    )
                payload = {
                    "group": key,
                    "variant": variant,
                    "mu": mu,
                    "summary": summarize(results),
                    "episodes": rows,
                    "predictions": predictions,
                    "contact_proofs": proofs,
                    "clip_results": [asdict(r) for r in results],
                    "wall_seconds": time.perf_counter() - tick,
                    "finished_utc": utc(),
                    "other_gpu_processes_at_end": gpu_others(),
                }
                save_json(path, payload)
                print(
                    json.dumps(
                        {
                            "group": key,
                            "summary": payload["summary"],
                            "seconds": payload["wall_seconds"],
                        }
                    ),
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
                update(state="group_saved", current_group=key)
            command = [sys.executable, str(HERE / "sweep_report.py"), "--eval-dir", str(target)]
            update(state="reporting_partial")
            subprocess.run(command, cwd=ROOT, check=True)
        all_rows = []
        for path in sorted(groups.glob("*.json")):
            all_rows += json.loads(path.read_text())["episodes"]
        write_csv(target / "episodes.csv", all_rows)
        assert state_hash(base) == frozen
        assert file_hash(ROOT / m["pretrained"]) == m["pretrained_sha256"]
        update(
            state="complete",
            finished_utc=utc(),
            episodes=len(all_rows),
            report=str((target / "analysis/RESULTS_ZH.md").relative_to(ROOT)),
            original_policy_unchanged=True,
        )
        save_json(
            OUT / "LATEST_EVALUATION.json",
            {
                "state": "complete",
                "directory": str(target.relative_to(ROOT)),
                "report": str((target / "analysis/RESULTS_ZH.md").relative_to(ROOT)),
            },
        )
    except BaseException as exc:
        update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
