"""Two short real-simulator interface checks; NOT a fine-tuned policy benchmark."""

import json
from dataclasses import asdict

import torch
from common import OUT, ROOT, save_json
from evaluate import chunk
from models import AdaptivePolicy

from supermimic.eval.benchmark import load_policy_from_checkpoint


def main():
    torch.set_num_threads(2)
    target = OUT / "evaluation_smoke.json"
    if target.exists():
        raise FileExistsError(target)
    base, cfg = load_policy_from_checkpoint(ROOT / "weights/realmimic_g1_86k.pt", "cuda:0")
    model = AdaptivePolicy(base, 0.1).to("cuda:0")
    scratch = torch.load(OUT / "validation/smoke_only.pt", map_location="cpu", weights_only=False)
    model.load_learned(scratch["learned"])
    model.residual_enabled = False
    path = ROOT / "data/npz/lafan1_seg10/walk1_subject1__seg000.npz"
    torch.manual_seed(0)
    br, bd, _, bp = chunk(base, [path], cfg, 0.3, "base", 0)
    torch.manual_seed(0)
    dr, dd, _, dp = chunk(model, [path], cfg, 0.3, "residual_disabled", 0)
    assert br[0].steps == 500 and dr[0].steps == 500
    assert bd[0]["initial_state_sha256"] == dd[0]["initial_state_sha256"]
    assert bd[0]["capacity_overflow_steps"] == 0 and dd[0]["capacity_overflow_steps"] == 0
    report = {
        "status": "passed",
        "scope": "Evaluation-interface smoke only: existing 10 s LAFAN1 segment at mu=.3; base vs residual-disabled scratch wrapper. Not full-clip benchmark or trained-adaptation result. No LAFAN1 optimization.",
        "base": asdict(br[0]),
        "residual_disabled": asdict(dr[0]),
        "diagnostics": bd + dd,
        "contact_proofs": bp + dp,
        "disabled_action_equivalence": "exact on every shared observation, checked inside evaluator",
        "paired_initial_states": True,
    }
    save_json(target, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
