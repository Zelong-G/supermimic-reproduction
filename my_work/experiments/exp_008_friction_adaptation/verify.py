"""Check saved paired trajectories, metric arithmetic, contacts and history alignment."""

import json

import numpy as np
from analyze import data, latest, pairing_audit
from run import HERE, OUT, save


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    audit = pairing_audit(cfg)
    checks = []
    for stage in ["fixed", "switch"]:
        cases = (
            cfg["cases"]
            if stage == "fixed"
            else [c for c in cfg["cases"] if c["category"] == "Walking"]
        )
        for case in cases:
            base = data(stage, case["name"], 1.0)
            for mu in cfg["frictions"] if stage == "fixed" else [1.0, 0.3, 0.15]:
                a = data(stage, case["name"], mu)
                n = len(a["mu"])
                meta = json.loads((OUT / stage / f"{case['name']}__mu{mu:g}.json").read_text())
                assert a["history"].shape == (n, 930) and a["goal"].shape == (n, 650)
                assert np.isfinite(a["states"]).all() and np.isfinite(a["history"]).all()
                current = latest(a["history"])
                np.testing.assert_allclose(current[1:, 64:], a["actions"][:-1], atol=1e-6)
                np.testing.assert_allclose(current[:, 29:58], a["states"][:, 42:], atol=1e-5)
                assert abs(meta["norm_return"] - a["reward"].sum() / (0.1 * case["frames"])) < 1e-10
                assert np.allclose(base["states"][0], a["states"][0], atol=1e-6, rtol=0)
                assert not a["terminated"][:-1].any() and not a["truncated"][:-1].any()
                hit = a["active_foot_contacts"] > 0
                assert hit.any()
                np.testing.assert_allclose(a["contact_mu_min"][hit], a["mu"][hit], atol=1e-6)
                np.testing.assert_allclose(a["contact_mu_max"][hit], a["mu"][hit], atol=1e-6)
                if stage == "switch":
                    if case["name"] in audit["eligible_clips"]:
                        for key in ["states", "history", "actions", "reward", "goal"]:
                            np.testing.assert_allclose(
                                a[key][:150], base[key][:150], atol=1e-4, rtol=0
                            )
                    np.testing.assert_allclose(a["sim_time"][150], 3.0, atol=1e-3)
                mask = a["overflow_mask"]
                values, counts = np.unique(mask, return_counts=True)
                checks.append(
                    {
                        "stage": stage,
                        "name": case["name"],
                        "mu": mu,
                        "steps": n,
                        "overflow_masks": {str(int(v)): int(c) for v, c in zip(values, counts)},
                        "capacity_errors": int(np.sum((mask & ~1536) != 0)),
                        "solver_limit_steps": int(np.sum((mask & 1536) != 0)),
                    }
                )
    assert not any(c["capacity_errors"] for c in checks), (
        "Capacity-affected trajectories require paired repair before interpretation"
    )
    save(
        OUT / "validation.json",
        {
            "status": "passed",
            "checks": checks,
            "switch_pairing_audit": audit,
            "notes": "Verified history last-action and joint-velocity alignment, paired initial states, actual active-contact friction, full-horizon score arithmetic, finite data, first-done stopping. Switch prefix gate applies to the eligible subset; attempted clips failing it remain recorded and are excluded from online-response inference. Nonzero bitmasks are classified, not assumed to imply lost contacts.",
        },
    )
    print(
        "Validated",
        len(checks),
        "episodes; capacity error steps:",
        sum(c["capacity_errors"] for c in checks),
    )


if __name__ == "__main__":
    main()
