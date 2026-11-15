> Latest status: the user completed 5,000 joint updates in tmux. Final checkpoint is joint_seed0/ckpt_0005000.pt. Full LAFAN1 paired evaluation is running; historical validation details below remain as an audit record.

# Experiment 009 — Implemented and validated; fine-tuning not started

The implementation is isolated under `my_work/experiments/exp_009_history_physics_adaptation/`.
The only requested index edit is `my_work/README.md`. Community source, original
checkpoint, datasets and Experiments 001–008 remain unchanged. No commit/push.

## Measured validation

| Check | Actual result |
|---|---|
| Existing Exp009 work | Absent before this implementation |
| CPU tests | 7 passed |
| Checkpoint | `weights/realmimic_g1_86k.pt`, iteration 86,000, strict Policy/WM/Critic loading |
| Original sizes | 4096 environments, batch 2048, horizon 32; preserved |
| Observation/goal | 930 /650; correct term-major current-frame extraction |
| Zero residual action difference | Exactly 0 on real pretrained inputs |
| Disabled residual | Exact base actions on identical inputs, before and after scratch learning |
| Trainable new parameters | 307,870; Encoder +Adapter jointly |
| World Model | Frozen and differentiable with respect to actions/states |
| Original Policy/WM/normalizers | Before/after full-update state hashes identical |
| Critic | Original architecture/GAE; updated and confirmed changed |
| Full-size scratch update | Passed, including original Critic update |
| PyTorch peak allocated | 15,645,487,616 bytes ≈14.57 GiB |
| PyTorch peak reserved | 16,382,951,424 bytes ≈15.26 GiB |
| Observed process memory | 18,606 MiB ≈18.17 GiB, `nvidia-smi` monitor |
| Full-update duration | 261.89 seconds, including first compilation; not steady-state throughput |
| OOM | None; no reduction of batch, horizon or environments |
| Checkpoint/optimizers | New modules and both optimizers round-trip exactly |
| Resume restore method | Executed on CPU with CUDA RNG setter mocked/checked; models, counters, RNG and sampler restored |
| Real evaluator interface | Two 500-step LAFAN1 segment episodes at μ=0.3 passed; real contact parameters verified |
| Formal 5,000-iteration fine-tuning | **Not started** |
| Formal full-clip LAFAN1 comparison | **Not started** |

The process memory monitor was attached during compilation and sampled once per
second; its peak is an observed lower bound. PyTorch peak statistics cover the
entire full update, while CUDA/Warp/context memory is not all included in those
PyTorch statistics. No steady-state training-throughput claim is made from one
compile-heavy validation update. Production logging measures every iteration.

## Gradient and loss calibration

Real BONES-SEED replay starts, B=8/H=4, were used for diagnostic gradients; those
diagnostic sizes never replace the full-training 2048/32/4096 configuration.

| Term/path | Measured value |
|---|---:|
| Initial tracking loss −J | -9.530531 |
| Initial friction MSE | 0.0856362 |
| Initial raw residual regularization | 0 |
| Tracking→Encoder gradient at zero final layer | 0 (expected) |
| Tracking→Adapter gradient at zero final layer | 0.0122268 |
| Friction→Encoder gradient | 0.341494 |
| Tracking→Encoder after temporary nonzero output | 2.8495e-6 |
| Tracking→WM next-state gradient | 1.03033 |
| WM next-state→Encoder gradient | 2.1107e-5 |

The temporary nonzero output weights are restored before the full scratch update.
This is not a separate training stage. Encoder supervision and Adapter tracking
updates begin together in one joint objective. The proxy verifies that all eight
diagnostic WM calls (two four-step rollouts) receive unchanged ground-truth ξ,
and every step recomputes μ̂ from its current history.

Defaults chosen after inspecting these scales/gradients: λμ=0.01,
λa=0.0038847475, α=0.1. Calibration targets bounded supervision/regularization
gradients near 10%/1% of the tracking Adapter gradient; the λμ lower bound binds.
These are provisional initial defaults, not LAFAN1-tuned or optimal weights.

Full-size first-update measurements: J=9.483576, friction MSE=0.0718737,
friction MAE=0.232597, joint loss=-9.482857; Encoder/Adapter gradient norms
0.0003360 /0.0095241. The measured residual magnitude and raw regularization are
zero on this first forward pass because the output layer starts at zero; both
new modules changed after the joint optimizer step. This is correctness evidence,
not evidence that one update improves tracking.

## Data and physics

Original `data/library/bones_paper.pt` is a truncated unreadable archive. Reuse
the already validated, read-only Exp007 rebuild of the same filtered BONES-SEED
library: 129,491 clips /47,345,789 frames. No data was downloaded or rebuilt.
All training and gradient/full-size validation use BONES-SEED. LAFAN1 is used
only for the short evaluator-interface check and future generalization evaluation.

The evaluator-interface check used an existing 10-second LAFAN1 segment, not
the full benchmark. Initial states match, and the residual-disabled actor equals
the base on every identical observation. In the latest check their normalized
returns were 0.9761433 and 0.9754406. Both completed 500 steps with zero capacity
errors; solver line-search flags (1024) occurred on 23 and 25 steps. These results
show a small numerical repeatability floor under the unchanged standard solver,
not a learned adaptation effect. All attempted technical records remain saved.

Main future evaluation preserves full clips and original benchmark physics:
35 contacts /250 constraints /10 solver /20 line-search iterations. It records
capacity versus solver-limit flags and verifies active foot-ground contact μ
every step. Numerical-control checks are needed before interpreting flagged
episodes or very small policy differences. Foot slip velocity/distance remains
unavailable until a contact-point estimator is validated; it is not invented.

## What has and has not been established

Established: architecture compatibility; exact initial/disabled action parity;
frozen-model invariance; required gradient routes; memory feasibility for one
full-size update; module/optimizer serialization; resume-state restoration logic;
and the real simulator evaluation interface.

Not established: improved low-friction robustness, preserved standard-friction
performance after 5,000 updates, accurate friction identification, or causal benefit
from history/friction conditioning. No second fine-tuning run or parameter-matched
trained ablation is performed. A long-running interruption/resume with actual
simulator/replay rebuilding is not yet exercised; as upstream, replay/simulator
state is refilled rather than checkpointed bitwise. Partial-iteration error
snapshots are diagnostic; resume the last completed checkpoint.

## Artifacts and manual next commands

`inspection.json`, `validation/full_check.json`, `validation/calibration.json`,
`audit.json`, `unit_tests.log`, `evaluation_smoke.json`, and complete logs provide
the evidence. `validation/smoke_only.pt` is a validation artifact, not a fine-tuned
model. Source hashes cover 140 protected source/documentation files, and the
pretrained file hash is unchanged. An existing Exp007 monitor-status modification
was already present before this task and was not edited here.

English diagnostic PNG/PDF figures are in `validation/analysis/`:
`validation_loss_and_gradients` and `validation_gpu_memory`. The report manifest
explicitly lists all seven fine-tuning/evaluation figure types as missing because
their measurements do not yet exist. `report.py` implements those figures for
the later real training and evaluation logs.

```bash
# Start the single 5,000-iteration run manually
bash my_work/experiments/exp_009_history_physics_adaptation/run.sh train --run-name joint_seed0

# After training, evaluate base /adapted /residual-disabled on existing full LAFAN1
OMP_NUM_THREADS=2 .venv/bin/python -u my_work/experiments/exp_009_history_physics_adaptation/evaluate.py \
  --checkpoint my_work/outputs/exp_009_history_physics_adaptation/joint_seed0/ckpt_0005000.pt \
  --run-name lafan1_joint5000

# Generate only figures supported by actual logs
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python my_work/experiments/exp_009_history_physics_adaptation/report.py \
  --run-dir my_work/outputs/exp_009_history_physics_adaptation/joint_seed0 \
  --eval-dir my_work/outputs/exp_009_history_physics_adaptation/evaluation/lafan1_joint5000
```
