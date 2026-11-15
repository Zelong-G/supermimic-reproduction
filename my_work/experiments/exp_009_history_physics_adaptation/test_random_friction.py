import numpy as np
from random_friction_protocol import clustered_interval, prefix_audit, schedule


def test_schedule_is_paired_and_survival_independent():
    a = schedule("walk1_subject1", 2, 1000)
    np.testing.assert_array_equal(a[:150], 1)
    np.testing.assert_array_equal(a[:400], schedule("walk1_subject1", 2, 400))
    assert np.all((a >= 0.2) & (a <= 1.5))
    assert np.all(a[150:300] == a[150])
    assert a[150] != a[300]
    assert not np.array_equal(a, schedule("walk1_subject1", 1, 1000))
    np.testing.assert_array_equal(schedule("walk1_subject1", 2, 1000, True), 1)


def test_pairing_gate_rejects_numerical_divergence():
    x = np.zeros((150, 10), dtype=np.float32)
    y = x.copy()
    y[149, 4] = 0.001
    assert not prefix_audit(x, y)["passed"]
    assert prefix_audit(x, x)["passed"]
    assert not prefix_audit(x, x[:100])["passed"]


def test_motion_cluster_interval_does_not_treat_frames_as_samples():
    lo, hi = clustered_interval([0.1, 0.1, 0.1])
    assert abs(lo - 0.1) < 1e-12 and abs(hi - 0.1) < 1e-12
