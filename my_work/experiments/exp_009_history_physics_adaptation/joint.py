"""Small extensions of upstream imagination: auxiliary outputs and one joint actor loss.

Original rollout, reward, termination, terminal critic objective and GAE are reused.
This local orchestration omits the upstream World Model optimizer entirely.
"""

from __future__ import annotations

import torch

from supermimic.imagination.compiled import ImagStep
from supermimic.imagination.objective import policy_objective
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.imagination.rollout import imagine
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major


def grad_norm(parameters):
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return torch.stack(values).sum().sqrt() if values else torch.tensor(0.0)


def prediction_samples(estimate, xi, valid):
    """Deterministic, aligned logging sample; never consumes training RNG."""
    pred = estimate.detach().reshape(-1)
    target = xi[:, 0][None].expand(estimate.shape[0], -1).reshape(-1)
    weight = valid.detach().reshape(-1)
    eligible = (weight > 1e-6).nonzero().flatten()
    ids = eligible[
        torch.linspace(0, len(eligible) - 1, min(256, len(eligible)), device=eligible.device).long()
    ]
    return [
        {"predicted_mu": p, "ground_truth_mu": t, "valid_weight": v}
        for p, t, v in zip(
            pred[ids].cpu().tolist(), target[ids].cpu().tolist(), weight[ids].cpu().tolist()
        )
    ]


class AdaptiveImagStep(ImagStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_trace()

    def reset_trace(self):
        self.mu = []
        self.raw = []
        self.delta = []

    def _step(self, s, hist, goal_ref, ref_anchor_pos, ref_anchor_quat, xi):
        g = self.goal(s, goal_ref, ref_anchor_pos, ref_anchor_quat)
        a, mu, raw, delta = self.policy.components(flatten_history_term_major(hist), g)
        # Only the original simulator context enters the frozen WM.
        nxt = self.wm.step(s, a, xi)
        obs = self.noisy(analytic_observation(nxt, a, xi, self.q0))
        return nxt, a, torch.cat([hist[1:], obs[None]], dim=0), g, mu, raw, delta

    def __call__(self, *args):
        nxt, a, hist, g, mu, raw, delta = self._fn(*args)
        # Trace storage is outside torch.compile, not a side effect during tracing.
        self.mu.append(mu)
        self.raw.append(raw)
        self.delta.append(delta)
        return nxt, a, hist, g


class JointTrainer(PolicyTrainer):
    def __init__(self, wm, policy, critic, lib, cfg, lambda_mu, lambda_action):
        super().__init__(wm, policy, critic, lib, cfg)
        self.opt = torch.optim.Adam(
            policy.trainable_parameters(), lr=cfg.lr, betas=cfg.betas, fused=True
        )
        self.step_fn = AdaptiveImagStep(
            wm,
            policy,
            lib.default_joint_pos,
            anchor_obs=cfg.anchor_obs,
            obs_noise=cfg.obs_noise,
            compile_mode=cfg.compile,
        )
        self.lambda_mu = lambda_mu
        self.lambda_action = lambda_action
        self.capture_predictions = False
        self.last_predictions = []
        self.wm.requires_grad_(False).eval()

    def losses(self, start):
        c = self.cfg
        self.step_fn.reset_trace()
        ro = imagine(
            self.lib,
            start,
            c.horizon,
            self.step_fn,
            self.heads_fn,
            c.future_frames,
            c.goal_features,
        )
        feat, goals, err = self._critic_inputs(ro)
        xi_norm = self.wm.spec.normalize_xi(start.xi)
        value = self.critic(feat[-1], goals[-1], xi_norm, err[-1]) if c.use_critic else None
        if value is not None and c.value_cap is not None:
            value = value.clamp(max=c.value_cap)
        J = policy_objective(ro, value, c.gamma)
        tracking = -J if c.objective_scale == "sum" else -J / c.horizon
        # Expected valid pre-action time: same survival convention as original objective.
        valid = ro.survival[:-1].detach()
        den = valid.sum().clamp_min(1.0)
        estimate = torch.stack(self.step_fn.mu).squeeze(-1)
        target = start.xi[:, 0][None].expand_as(estimate)
        friction = ((estimate - target).square() * valid).sum() / den
        mae = ((estimate - target).abs() * valid).sum() / den
        raw = torch.stack(self.step_fn.raw)
        delta = torch.stack(self.step_fn.delta)
        regularization = (raw.square().sum(-1) * valid).sum() / den
        magnitude = (delta.abs().mean(-1) * valid).sum() / den
        total = tracking + self.lambda_mu * friction + self.lambda_action * regularization
        metrics = {
            "imag_objective": J.detach(),
            "friction_mse": friction.detach(),
            "friction_mae": mae.detach(),
            "action_regularization": regularization.detach(),
            "total_loss": total.detach(),
            "residual_abs_mean": magnitude.detach(),
            "imag_return_undisc": (ro.survival[:-1] * ro.rewards).sum(0).mean().detach(),
            "imag_reward_mean": ro.rewards.mean().detach(),
            "imag_survival_H": ro.survival[-1].mean().detach(),
            "valid_time_weight": den.detach(),
        }
        return total, (tracking, friction, regularization), metrics, (ro, feat, goals, err, xi_norm)

    def _fit_critic(self, fits):
        c = self.cfg
        if not c.use_critic:
            return {}
        self.critic.requires_grad_(True)
        feat, goal, xi, err, target, values = (torch.cat(t) for t in zip(*fits))
        n = len(target)
        mb = n // c.critic_minibatches
        for _ in range(c.critic_epochs):
            perm = torch.randperm(n, device=target.device)
            for i in range(c.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                pred = self.critic(feat[idx], goal[idx], xi[idx], err[idx])
                loss = torch.nn.functional.mse_loss(pred, target[idx])
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c.critic_grad_clip)
                self.critic_opt.step()
        return {
            "critic_loss": float(loss.detach()),
            "critic_value_mean": float(values.mean()),
            "critic_target_mean": float(target.mean()),
        }

    def update(self, buffer, xi_norm_fn=None):
        c = self.cfg
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        self.opt.zero_grad(set_to_none=True)
        fits = []
        metrics = {}
        k = max(1, c.grad_accumulation)
        extra = (
            buffer.frame + c.start_margin <= self.lib.last_frame_of(buffer.frame)
            if c.start_margin > 0
            else None
        )
        for _ in range(k):
            start = buffer.sample_starts(c.batch_size, c.history, extra_mask=extra)
            loss, _, m, fit = self.losses(start)
            if self.capture_predictions:
                self.last_predictions = prediction_samples(
                    torch.stack(self.step_fn.mu), start.xi, fit[0].survival[:-1]
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nonfinite joint loss; stopped without changing batch/horizon/envs"
                )
            (loss / k).backward()
            for name, v in m.items():
                metrics[name] = metrics.get(name, 0.0) + float(v) / k
            if c.use_critic:
                fits.append(self._critic_data(*fit))
            self.step_fn.reset_trace()
            del loss, fit
        metrics["encoder_grad_norm"] = float(grad_norm(self.policy.encoder.parameters()))
        metrics["adapter_grad_norm"] = float(grad_norm(self.policy.adapter.parameters()))
        params = self.policy.trainable_parameters()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError("Nonfinite gradient")
        metrics["joint_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, c.grad_clip_norm))
        torch.nn.utils.clip_grad_value_(params, c.grad_clip_value)
        self.opt.step()
        metrics.update(self._fit_critic(fits))
        self.wm.requires_grad_(False)
        return metrics
