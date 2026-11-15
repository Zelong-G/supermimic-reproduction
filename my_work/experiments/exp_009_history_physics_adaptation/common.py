"""Experiment-local IO, immutable-source audits and checkpoint-based configuration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "my_work/outputs/exp_009_history_physics_adaptation"
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")
# Torch's generated compilation cache is also confined to Experiment 009.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(OUT / ".cache/inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(OUT / ".cache/triton"))


def settings():
    return json.loads((HERE / "config.json").read_text())


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def output_path(path):
    p = Path(path).resolve()
    if not p.is_relative_to(OUT.resolve()):
        raise ValueError(f"Output escapes Experiment009: {p}")
    return p


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for key, t in sorted(model.state_dict().items()):
        v = t.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def protected_sources():
    paths = []
    for folder in ["src", "scripts", "tools", "my_work/experiments"]:
        paths.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file()
            and ".git" not in p.parts
            and "__pycache__" not in p.parts
            and HERE not in p.parents
        )
    # Prior result files can be large; code audit is bytewise, outputs use size/mtime.
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def build_config(ck, run_name):
    from supermimic.presets import release

    cfg = release()

    def assign(obj, d):
        for k, v in d.items():
            if not hasattr(obj, k):
                raise ValueError(f"Unsupported checkpoint field: {type(obj).__name__}.{k}")
            old = getattr(obj, k)
            if dataclasses.is_dataclass(old):
                assign(old, v)
            else:
                setattr(obj, k, tuple(v) if isinstance(old, tuple) else v)

    assign(cfg, ck["config"])
    s = settings()
    cfg.data.library = ROOT / s["library"]
    cfg.data.npz_dir = ROOT / "data/npz/bones_seed"
    cfg.log_root = OUT
    cfg.run_name = run_name
    cfg.seed = s["seed"]
    cfg.resume = None
    cfg.init_from = ROOT / s["pretrained"]
    cfg.schedule.total_iterations = s["iterations"]
    cfg.schedule.wm_warmup_iterations = 0
    cfg.schedule.checkpoint_interval = s["checkpoint_interval"]
    assert cfg.env.num_envs == 4096 and cfg.policy_trainer.batch_size == 2048
    assert cfg.policy_trainer.horizon == 32 and cfg.policy_trainer.history == 10
    assert cfg.policy.history == 10 and cfg.amp == "none"
    assert cfg.env.domain_randomization and cfg.env.sampling_mode == "return"
    return cfg
