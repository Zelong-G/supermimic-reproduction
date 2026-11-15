"""Time-based continuation must bypass 5k without changing training sizes or input config."""

from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
from train import configure_continuation


def objects():
    exp = NS(
        iteration=5000,
        cfg=NS(
            schedule=NS(total_iterations=5000, max_hours=11),
            policy_trainer=NS(horizon=32, batch_size=2048),
            env=NS(num_envs=4096),
        ),
    )
    ck = {"config": {"schedule": {"total_iterations": 5000, "max_hours": 11}}}
    return exp, ck


def test_six_hour_branch_preserves_parent_and_sizes():
    exp, ck = objects()
    before = deepcopy(ck)
    control = configure_continuation(exp, ck, extend_hours=6)
    assert exp.cfg.schedule.total_iterations > 100000
    assert exp.cfg.schedule.max_hours == 6
    assert control["training_sizes"] == {"horizon": 32, "batch": 2048, "envs": 4096}
    assert ck == before and exp.iteration == 5000


def test_resume_time_branch_keeps_saved_ceiling():
    exp, ck = objects()
    exp.iteration = 80000
    ck["config"]["schedule"].update(total_iterations=1_000_000_000, max_hours=6.0)
    configure_continuation(exp, ck)
    assert exp.cfg.schedule.total_iterations == 1_000_000_000 and exp.cfg.schedule.max_hours == 6


def test_old_completed_run_needs_explicit_extension():
    exp, ck = objects()
    with pytest.raises(ValueError):
        configure_continuation(exp, ck)


def test_ambiguous_limits_are_rejected():
    exp, ck = objects()
    with pytest.raises(ValueError):
        configure_continuation(exp, ck, extend_hours=6, total_iterations=20000)
