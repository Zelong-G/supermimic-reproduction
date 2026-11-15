"""CPU-only artifact, frozen-source and learned-checkpoint/optimizer roundtrip audit."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch
from common import HERE, OUT, ROOT, file_hash, protected_sources, save_json
from models import AdaptivePolicy

from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.policy.nets import Critic, CriticCfg
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


def main():
    torch.set_num_threads(2)
    inspection = json.loads((OUT / "inspection.json").read_text())
    current = protected_sources()
    assert current == inspection["source_hashes"], "Protected community/Exp001–008 source changed"
    assert file_hash(ROOT / inspection["checkpoint"]) == inspection["checkpoint_sha256"]
    check = json.loads((OUT / "validation/full_check.json").read_text())
    assert check["status"] == "passed"
    assert check["full_size"] == {"horizon": 32, "batch": 2048, "envs": 4096}
    ck = torch.load(OUT / "validation/smoke_only.pt", map_location="cpu", weights_only=False)
    base, cfg = load_policy_from_checkpoint(ROOT / ck["pretrained"], "cpu")
    model = AdaptivePolicy(base, ck["settings"]["alpha"])
    model.load_learned(ck["learned"])
    assert all(
        torch.equal(v, ck["learned"]["encoder"][k]) for k, v in model.encoder.state_dict().items()
    )
    assert all(
        torch.equal(v, ck["learned"]["adapter"][k]) for k, v in model.adapter.state_dict().items()
    )
    actor_opt = torch.optim.Adam(model.trainable_parameters(), lr=cfg["policy_trainer"]["lr"])
    actor_opt.load_state_dict(ck["joint_opt"])
    spec = StateSpec(StateSpecCfg(**cfg["world_model"]["state_spec"]), "cpu")
    critic = Critic(
        CriticCfg(**cfg["critic"]), feat_dim=spec.feat_dim, pcfg=base.cfg, state_frame=spec.frame
    )
    critic.load_state_dict(ck["critic"], strict=True)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg["policy_trainer"]["critic_lr"])
    critic_opt.load_state_dict(ck["critic_opt"])

    def compare(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(compare(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(compare(x, y) for x, y in zip(a, b))
        return a == b

    assert compare(actor_opt.state_dict(), ck["joint_opt"]) and compare(
        critic_opt.state_dict(), ck["critic_opt"]
    )
    from train import Experiment

    from realmimic.export.web_world_model import load_world_model

    resumed = Experiment.__new__(Experiment)
    resumed.device = "cpu"
    resumed.policy = model
    resumed.critic = critic
    resumed.wm = load_world_model(ROOT / ck["pretrained"])[0].requires_grad_(False)
    resumed.frozen_hashes = ck["frozen_hashes"]
    resumed.policy_trainer = SimpleNamespace(opt=actor_opt, critic_opt=critic_opt)
    resumed.command = SimpleNamespace(
        **{k: torch.zeros_like(v) for k, v in ck["sampler"].items()},
        _bin_count_synced=torch.zeros_like(ck["sampler"]["bin_count"]),
    )
    with patch("torch.cuda.set_rng_state_all") as cuda_restore:
        resumed.resume_experiment(OUT / "validation/smoke_only.pt")
        cuda_restore.assert_called_once()
    assert resumed.iteration == 1 and resumed.history == ck["metrics"]
    assert torch.equal(torch.get_rng_state(), ck["rng_torch"])
    assert all(torch.equal(getattr(resumed.command, k), v) for k, v in ck["sampler"].items())
    original = torch.load(ROOT / ck["pretrained"], map_location="cpu", weights_only=False)
    critic_changed = any(not torch.equal(v, original["critic"][k]) for k, v in ck["critic"].items())
    assert critic_changed
    with torch.no_grad():
        model.residual_enabled = False
        h = torch.randn(3, 930)
        g = torch.randn(3, 650)
        assert torch.equal(model(h, g), model.base(h, g))
    save_json(
        OUT / "audit.json",
        {
            "status": "passed",
            "protected_sources_unchanged": True,
            "protected_file_count": len(current),
            "pretrained_file_unchanged": True,
            "learned_module_roundtrip_exact": True,
            "joint_optimizer_roundtrip_exact": True,
            "critic_optimizer_roundtrip_exact": True,
            "critic_updated_confirmed": critic_changed,
            "resume_restore_method_cpu_checked": True,
            "cuda_rng_restore_call_checked_without_gpu_launch": True,
            "trained_residual_disabled_action_equivalence_exact": True,
            "full_size_validation_passed": True,
            "fine_tuning_started": False,
            "experiment_code_sha256": {p.name: file_hash(p) for p in HERE.glob("*.py")},
            "experiment_configuration_sha256": file_hash(HERE / "config.json"),
            "evaluation_interface_smoke_status": json.loads(
                (OUT / "evaluation_smoke.json").read_text()
            )["status"],
            "preexisting_dirty_file": "my_work/outputs/exp_007_full_training/full_seed0/monitor_status.json (present before this task; not edited here)",
            "allowed_index_edit": "my_work/README.md",
        },
    )
    print(
        "Audit passed: protected sources, original checkpoint, learned modules, both optimizers, Critic update and disabled-action equivalence."
    )


if __name__ == "__main__":
    main()
