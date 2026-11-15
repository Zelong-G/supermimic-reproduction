"""CPU invariants; real pretrained/WM/full-size gradient checks live in validate.py."""

import torch
from models import AdaptivePolicy, current_observation

from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major


def model():
    return AdaptivePolicy(
        Policy(
            PolicyCfg(
                encoder_hidden=(32,),
                decoder_hidden=(32,),
                goal_dim=650,
                goal_frame_dim=59,
                goal_features="sonic",
            )
        )
    )


def test_term_major_current():
    h = torch.arange(10 * 2 * 93, dtype=torch.float32).reshape(10, 2, 93)
    flat = flatten_history_term_major(h)
    torch.testing.assert_close(current_observation(flat), h[-1], atol=0, rtol=0)
    assert not torch.equal(flat[:, -93:], h[-1])


def test_zero_and_disabled_residual():
    p = model()
    h = torch.randn(3, 930)
    g = torch.randn(3, 650)
    assert torch.equal(p(h, g), p.base(h, g))
    with torch.no_grad():
        p.adapter[-1].bias.fill_(1)
    assert not torch.equal(p(h, g), p.base(h, g))
    p.residual_enabled = False
    assert torch.equal(p(h, g), p.base(h, g))


def test_bounds_and_freeze():
    p = model().train()
    h = torch.randn(4, 930)
    g = torch.randn(4, 650)
    a, mu, raw, delta = p.components(h, g)
    assert a.shape == (4, 29) and mu.shape == (4, 1) and raw.shape == (4, 29)
    assert ((mu >= 0.3) & (mu <= 1.2)).all() and (delta.abs() <= 0.1).all()
    assert not p.base.training and all(not v.requires_grad for v in p.base.parameters())
    assert sum(v.numel() for v in p.trainable_parameters()) == 307870


def test_friction_is_not_an_observation_input():
    s = torch.randn(3, 71)
    s[:, 3:7] = torch.tensor([1.0, 0, 0, 0])
    a = torch.randn(3, 29)
    xi = torch.zeros(3, 33)
    first = analytic_observation(s, a, xi, torch.zeros(29))
    xi[:, 0] = 0.99
    assert torch.equal(first, analytic_observation(s, a, xi, torch.zeros(29)))


def test_encoder_tracking_gate_and_supervision():
    p = model()
    h = torch.randn(4, 930)
    g = torch.randn(4, 650)
    p(h, g).sum().backward()
    assert all(v.grad is None or torch.count_nonzero(v.grad) == 0 for v in p.encoder.parameters())
    assert p.adapter[-1].weight.grad.abs().sum() > 0
    p.zero_grad(set_to_none=True)
    with torch.no_grad():
        p.adapter[-1].weight.normal_(0, 0.01)
    p(h, g).sum().backward()
    assert sum(v.grad.abs().sum() for v in p.encoder.parameters() if v.grad is not None) > 0
    assert all(v.grad is None for v in p.base.parameters())
    p.zero_grad(set_to_none=True)
    (_, mu, _, _) = p.components(h, g)
    (mu - 0.4).square().mean().backward()
    assert sum(v.grad.abs().sum() for v in p.encoder.parameters() if v.grad is not None) > 0


def test_policy_normalizers_do_not_update():
    from common import state_hash

    p = model()
    before = state_hash(p.base)
    for _ in range(3):
        p(torch.randn(4, 930), torch.randn(4, 650))
    assert state_hash(p.base) == before


def test_prediction_logging_alignment_and_rng():
    from joint import prediction_samples

    xi = torch.zeros(2048, 33)
    xi[:, 0] = torch.linspace(0.3, 1.2, 2048)
    pred = xi[:, 0][None, :, None].expand(32, -1, 1).clone()
    valid = torch.ones(32, 2048)
    valid[0] = 0
    rng = torch.get_rng_state().clone()
    samples = prediction_samples(pred, xi, valid)
    assert len(samples) == 256 and torch.equal(rng, torch.get_rng_state())
    assert all(r["predicted_mu"] == r["ground_truth_mu"] and r["valid_weight"] > 0 for r in samples)
