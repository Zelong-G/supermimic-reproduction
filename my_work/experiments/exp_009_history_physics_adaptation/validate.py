"""Real BONES-SEED validation; one scratch update at the unmodified release sizes."""

from __future__ import annotations

import dataclasses
import gc
import json
import time

import torch
from common import OUT, ROOT, protected_sources, save_json, settings
from joint import JointTrainer


def norms(loss, model):
    ep = list(model.encoder.parameters())
    ap = list(model.adapter.parameters())
    grads = torch.autograd.grad(loss, ep + ap, retain_graph=True, allow_unused=True)

    def norm(gs):
        return float(
            sum(
                (g.square().sum() for g in gs if g is not None), torch.zeros((), device=loss.device)
            ).sqrt()
        )

    return {"encoder": norm(grads[: len(ep)]), "adapter": norm(grads[len(ep) :])}


class ContextAuditWM(torch.nn.Module):
    def __init__(self, wm, xi):
        super().__init__()
        self.inner = wm
        self.expected = xi
        self.calls = 0
        self.outputs = []

    @property
    def spec(self):
        return self.inner.spec

    @property
    def heads(self):
        return self.inner.heads

    def termination_logits(self, *args):
        return self.inner.termination_logits(*args)

    def step(self, s, a, xi):
        torch.testing.assert_close(xi, self.expected, rtol=0, atol=0)
        self.calls += 1
        nxt = self.inner.step(s, a, xi)
        self.outputs.append(nxt)
        return nxt


def validate(name):
    from train import Experiment

    out = OUT / name
    if out.exists():
        raise FileExistsError(out)
    before = protected_sources()
    exp = None
    report = {
        "status": "running",
        "full_training_started": False,
        "full_size": {"horizon": 32, "batch": 2048, "envs": 4096},
    }
    try:
        exp = Experiment(name)
        print("Loaded release checkpoint and full BONES-SEED environment", flush=True)
        ck = torch.load(ROOT / settings()["pretrained"], map_location="cpu", weights_only=False)
        for label, module in [("policy", exp.policy.base), ("wm", exp.wm), ("critic", exp.critic)]:
            assert set(module.state_dict()) == set(ck[label])
            assert all(
                torch.equal(v.cpu(), ck[label][k].cpu()) for k, v in module.state_dict().items()
            ), label
        del ck
        h, g = exp._obs["policy"][:8], exp._obs["goal"][:8]
        with torch.no_grad():
            base = exp.policy.base(h, g)
            a, mu, _raw, delta = exp.policy.components(h, g)
            assert torch.equal(base, a) and torch.count_nonzero(delta) == 0
            assert h.shape == (8, 930) and g.shape == (8, 650) and mu.shape == (8, 1)
        report["zero_residual_max_action_difference"] = float((base - a).abs().max())
        exp.fill_replay()
        print("Replay filled using original 4096 environments / 40 collection steps", flush=True)
        start = exp.buffer.sample_starts(8, 10)
        from supermimic.world_model.heads import analytic_observation

        xi2 = start.xi.clone()
        xi2[:, 0] = 1.2 - xi2[:, 0]
        torch.testing.assert_close(
            analytic_observation(
                start.state, start.prev_action, start.xi, exp.lib.default_joint_pos
            ),
            analytic_observation(start.state, start.prev_action, xi2, exp.lib.default_joint_pos),
            atol=0,
            rtol=0,
        )
        audit = ContextAuditWM(exp.wm, start.xi)
        cfg = dataclasses.replace(exp.cfg.policy_trainer, horizon=4, batch_size=8, compile=None)
        probe = JointTrainer(audit, exp.policy, exp.critic, exp.lib, cfg, 1.0, 1.0)
        exp.critic.requires_grad_(False)
        initial = {
            k: {n: t.detach().clone() for n, t in d.items()}
            for k, d in exp.policy.learned_state().items()
        }
        total, terms, metrics, fit = probe.losses(start)
        initial_scales = {
            name: {"value": float(loss.detach()), "gradients": norms(loss, exp.policy)}
            for name, loss in zip(["tracking", "friction", "raw_residual_regularization"], terms)
        }
        assert initial_scales["tracking"]["gradients"]["adapter"] > 0
        assert initial_scales["tracking"]["gradients"]["encoder"] == 0, (
            "Zero output layer correctly blocks initial tracking-to-encoder gradient"
        )
        assert initial_scales["friction"]["gradients"]["encoder"] > 0
        del total, terms, metrics, fit
        probe.step_fn.reset_trace()
        audit.outputs.clear()
        with torch.no_grad():
            exp.policy.adapter[-1].weight.normal_(0, 0.001)
        total, terms, metrics, fit = probe.losses(start)
        scales = {
            name: {"value": float(loss.detach()), "gradients": norms(loss, exp.policy)}
            for name, loss in zip(["tracking", "friction", "raw_residual_regularization"], terms)
        }
        assert scales["tracking"]["gradients"]["encoder"] > 0
        wm_reward_grad = torch.autograd.grad(terms[0], audit.outputs[-1], retain_graph=True)[0]
        state_to_encoder = norms(audit.outputs[-1].square().mean(), exp.policy)["encoder"]
        assert wm_reward_grad.norm() > 0 and state_to_encoder > 0
        assert audit.calls == 8 and len(probe.step_fn.mu) == 4
        tracking_a = scales["tracking"]["gradients"]["adapter"]
        friction_e = scales["friction"]["gradients"]["encoder"]
        reg_a = scales["raw_residual_regularization"]["gradients"]["adapter"]
        lm = float(min(10.0, max(0.01, 0.1 * tracking_a / max(friction_e, 1e-12))))
        la = float(min(1.0, max(1e-5, 0.01 * tracking_a / max(reg_a, 1e-12))))
        calibration = {
            "lambda_mu": lm,
            "lambda_action": la,
            "basis": "Four-step, eight-start real BONES-SEED gradient diagnostic, never LAFAN1. Target supervised encoder gradient about 10% of tracking adapter gradient; raw-residual regularizer about 1%, bounded mu weight [0.01,10], action weight [1e-5,1]. Fixed thereafter; provisional defaults rather than reward-tuned hyperparameters.",
            "initial_zero_adapter": initial_scales,
            "temporary_nonzero_output_layer": scales,
            "tracking_to_wm_next_gradient_norm": float(wm_reward_grad.norm()),
            "wm_next_to_encoder_gradient_norm": state_to_encoder,
            "context_audit_calls": audit.calls,
            "encoder_recomputed_steps": len(probe.step_fn.mu),
        }
        save_json(out / "calibration.json", calibration)
        exp.policy.load_learned(initial)
        # Disabled residual must bypass even a nonzero adapter exactly.
        with torch.no_grad():
            exp.policy.adapter[-1].bias.fill_(0.2)
            exp.policy.residual_enabled = False
            assert torch.equal(exp.policy(h, g), exp.policy.base(h, g))
        exp.policy.load_learned(initial)
        exp.policy.residual_enabled = True
        del total, terms, metrics, fit, probe, audit, wm_reward_grad
        exp.policy_trainer.lambda_mu = lm
        exp.policy_trainer.lambda_action = la
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(
            f"Gradient checks passed; joint weights mu={lm:.6g}, action={la:.6g}. Starting one B=2048,H=32 update.",
            flush=True,
        )
        tick = time.perf_counter()
        full = exp.policy_trainer.update(exp.buffer)
        torch.cuda.synchronize()
        full["seconds"] = time.perf_counter() - tick
        full["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        full["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        assert full["encoder_grad_norm"] > 0 and full["adapter_grad_norm"] > 0
        assert all(torch.isfinite(torch.tensor(v)) for v in full.values())
        assert any(
            not torch.equal(t, initial["encoder"][k])
            for k, t in exp.policy.encoder.state_dict().items()
        )
        assert any(
            not torch.equal(t, initial["adapter"][k])
            for k, t in exp.policy.adapter.state_dict().items()
        )
        frozen = exp.assert_frozen()
        exp.iteration = 1
        exp.history = [full]
        # This checkpoint is a validation artifact, not a 5,000-iteration fine-tuned model.
        smoke = exp.save_experiment("smoke_only.pt")
        saved = torch.load(smoke, map_location="cpu", weights_only=False)
        assert saved["iteration"] == 1 and saved["frozen_hashes"] == frozen
        report.update(
            status="passed",
            pretrained_strict_load=True,
            zero_residual_exact=True,
            disabled_residual_exact=True,
            ground_truth_context_only_to_wm_and_supervision=True,
            full_update=full,
            calibration=calibration,
            frozen_hashes=frozen,
            trainable_parameters=sum(p.numel() for p in exp.policy.trainable_parameters()),
            checkpoint_roundtrip=True,
            smoke_checkpoint=str(smoke.relative_to(ROOT)),
            critic_updated=True,
            compile_mode=exp.cfg.policy_trainer.compile,
        )
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        report.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            stage="constructor or diagnostics"
            if exp is None
            else "see validation log for most recent stage",
        )
        raise
    finally:
        after = protected_sources()
        report["protected_source_hashes_unchanged"] = before == after
        if before != after:
            report["status"] = "failed"
            report["audit_error"] = "Protected source files changed"
        save_json(out / "full_check.json", report)
        if exp is not None:
            if exp.writer:
                exp.writer.close()
            exp.env.close()
