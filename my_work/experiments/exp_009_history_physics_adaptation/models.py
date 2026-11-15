"""History-only friction estimation and a bounded action residual; upstream models unchanged."""

from __future__ import annotations

import torch
from torch import nn

from supermimic.world_model.heads import OBS_TERMS


def current_observation(history):
    """Term-major [B,930] -> most recent [B,93], with gradient-preserving slices."""
    if history.shape[-1] != 930:
        raise ValueError("Expected original ten-frame, term-major history")
    pieces = []
    offset = 0
    for _, dim in OBS_TERMS:
        pieces.append(
            history[..., offset : offset + 10 * dim].reshape(*history.shape[:-1], 10, dim)[
                ..., -1, :
            ]
        )
        offset += 10 * dim
    return torch.cat(pieces, dim=-1)


class PhysicsEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(930, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)
        )

    def forward(self, history):
        return 0.3 + 0.9 * torch.sigmoid(super().forward(history))


class ResidualAdapter(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(126, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 29)
        )
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


class AdaptivePolicy(nn.Module):
    def __init__(self, base, alpha=0.1):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.cfg = base.cfg
        self.encoder = PhysicsEncoder()
        self.adapter = ResidualAdapter()
        self.alpha = alpha
        self.residual_enabled = True

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def normalized_history(self, history):
        d = self.base.decoder
        return d.norm(history) if d.norm is not None else history * d.scale

    def components(self, obs_history, goal):
        # Never no_grad here: gradients through frozen base inputs are part of BPTT.
        z = self.base.encoders["motion"](goal)
        base_action = self.base.decoder(z, obs_history)
        normalized = self.normalized_history(obs_history)
        mu = self.encoder(normalized)
        raw = self.adapter(torch.cat([current_observation(normalized), z, mu], dim=-1))
        delta = self.alpha * torch.tanh(raw)
        action = base_action + delta if self.residual_enabled else base_action
        return action, mu, raw, delta

    def forward(self, obs_history, goal, interface="motion"):
        if interface != "motion":
            raise ValueError("Only original motion interface supported")
        return self.components(obs_history, goal)[0]

    def trainable_parameters(self):
        return list(self.encoder.parameters()) + list(self.adapter.parameters())

    def learned_state(self):
        return {"encoder": self.encoder.state_dict(), "adapter": self.adapter.state_dict()}

    def load_learned(self, state):
        self.encoder.load_state_dict(state["encoder"], strict=True)
        self.adapter.load_state_dict(state["adapter"], strict=True)
