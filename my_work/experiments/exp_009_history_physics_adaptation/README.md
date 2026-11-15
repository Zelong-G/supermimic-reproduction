# Experiment 009 — History-Based Physics Adaptation

Isolated extension of community **realmimic / SuperMimic**. Preserve the repository
`LICENSE`, `NOTICE`, `LICENSES/`, and `CITATION.cff`; upstream implementation and
pretrained models remain credited to their original authors. No community files
or Experiments 001–008 are edited. The explicitly requested workspace index
`my_work/README.md` is the only change outside Experiment009 directories.

**Status:** implementation and validation only. The 5,000-iteration run is manual.
No claim about improved robustness is made before real fine-tuning/evaluation.
Consult `my_work/outputs/exp_009_history_physics_adaptation/STATUS.md` for measured
validation results and any limitations, rather than assuming the checks passed.

## Architecture and frozen models

- `weights/realmimic_g1_86k.pt` supplies original Policy, World Model, Critic and
  their architecture/configuration. Parent community `Trainer` performs strict
  model loading and builds unchanged analytic reward heads and physics.
- Original policy and motion encoder weights **and all normalizer buffers** stay
  frozen. The collection `act()` override never calls `update_input_norms`.
  Frozen base forwards are not wrapped in no_grad inside imagination: gradients
  through observations/goals must still reach earlier imagined states.
- New Physics Encoder: 930→256 ELU→128 ELU→1, with `0.3+0.9*sigmoid` output.
- New Residual Adapter: (93 current obs +32 frozen motion latent +1 estimate)
  →128 ELU→128 ELU→29, zero final layer; correction `0.1*tanh(raw)`.
- New modules reuse **frozen pretrained observation normalization**. Current obs
  is extracted term by term at history index -1, never by taking the final 93
  entries. This normalization is an explicit input preprocessing choice, not
  an update of the original policy's statistics.
- New trainable parameter count: 307,870. No privileged ξ or simulator state enters
  `AdaptivePolicy`, `PhysicsEncoder`, or `ResidualAdapter`. Original reference goals
  remain unchanged. The estimate is never detached before the Adapter.
- World Model parameters and normalizers are frozen; action/state derivatives
  remain enabled. Every imagined step uses **ground-truth** replay `xi`, not μ̂.
- Original Critic architecture, bootstrap, value cap, GAE γ/λ, epochs/minibatches
  and optimizer behavior are retained; **Critic is updated** after each actor update.

## Preserved training recipe

`common.build_config()` overlays every checkpoint field on the release dataclasses
and fails on unknown fields. Mandatory assertions enforce **4096 environments,
batch=2048, horizon=32, history=10**, fp32 and return-based reset sampling. Physics
step 0.005×4=0.02 s, PD/action scaling, noise, DR including foot friction [0.3,1.2],
replay capacity, motion sampling, future goals, rewards and GAE come from the
release. There is no silent size/horizon/environment reduction on OOM.

The fine-tuning counter starts at zero, with the original 2e-4 learning rate and
69,000-iteration cosine schedule, targeting 5,000 joint updates. No original
World Model update/warm-up occurs and no WM optimizer is retained. Original
40-step simulator replay fill is retained (it is not World Model training).

Use the original filtered **BONES-SEED** training motion library only. The archive
`data/library/bones_paper.pt` is truncated/unreadable; the configured read-only
Exp007 rebuild has the same validated 129,491 clips /47,345,789 frames. Both
archives remain untouched. No new data is downloaded. LAFAN1 is evaluation-only.
Reset-sampling strategy remains original; pretrained release lacks sampler
statistics, so fine-tuning starts with the community sampler priors.

## Joint objective and calibration

`JointTrainer` subclasses community `PolicyTrainer`; community `imagine`,
`ImagHeads`, `policy_objective`, `_critic_inputs`, `_critic_data` and GAE remain
in use. `AdaptiveImagStep` adds auxiliary tensor outputs outside compiled trace
storage; it otherwise uses original goal/noise/history/WM transitions.

One optimizer updates Encoder and Adapter together from the beginning:

```
L = original tracking loss + λμ · friction MSE + λa · raw-residual regularization
tracking loss = −J                  # original objective_scale=sum
raw-residual regularization = E[sum_j raw_j²]
action correction = α tanh(raw)     # normalized PD-command units, not torque
```

Auxiliary losses average across time/batch using detached pre-action survival
weights (`ro.survival[:-1]`), the same valid-time convention as the original
objective. Ground-truth friction is the constant episode context `start.xi[:,0]`;
μ̂ is recomputed from the shifted observation history on **every** step. Auxiliary
gradients can also reach earlier actions through future observations, as expected
in joint differentiable training.

`validation/calibration.json` records measured values and separate gradient norms
for all three terms, initially and with a temporary nonzero output layer.
Default λμ=0.01 and λa≈0.00388475 were chosen from those BONES-SEED diagnostic
gradients, using bounded targets of 10%/1% of the tracking Adapter gradient.
The supervision lower bound binds in this diagnostic; these are provisional
initial defaults, not tuned optimal weights. `config.json` can explicitly set them.
The temporary diagnostic weights are restored before the full-size scratch update;
there is **no Encoder pretraining stage** and no separate Adapter fine-tuning run.

At zero output initialization, the tracking→Encoder gradient is mathematically
zero. Supervision trains the Encoder immediately and tracking trains the Adapter;
after nonzero output weights, tracking gradients also reach the Encoder. The test
explicitly checks this expected gate instead of misreporting it as a broken graph.

## Commands (repository root)

To continue the completed 5k checkpoint in a separate branch for up to six hours:

```bash
bash my_work/experiments/exp_009_history_physics_adaptation/run.sh train \
  --resume my_work/outputs/exp_009_history_physics_adaptation/joint_seed0/ckpt_0005000.pt \
  --run-name joint_6h_from5000 --extend-hours 6
```

This restores Encoder, Adapter, Critic, optimizers and sampler state; it preserves
the parent directory and shared config. The old 5k iteration limit is bypassed
using a large safety ceiling; the invocation deadline controls stopping. Time is
counted from argument parsing, including subsequent model/environment startup;
module imports and completing the last iteration/checkpoint can add a small
overrun. The original learning-rate schedule continues from iteration 5000.
New outputs are in `joint_6h_from5000/`. A starting snapshot at iteration 5000
and periodic/final checkpoints are saved there. No new training is started merely
by adding this interface. To resume this branch later, use its checkpoint with
ordinary `--resume`; its saved limits are retained.

```bash
# CPU invariants and strict input/output inspection
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  my_work/experiments/exp_009_history_physics_adaptation/test_models.py
bash my_work/experiments/exp_009_history_physics_adaptation/run.sh inspect

# One validation-only update at full original sizes; can take time to compile.
# Refuses existing output directory. Use --run-name validation_002 for a new audit.
bash my_work/experiments/exp_009_history_physics_adaptation/run.sh validate

# MANUAL: the single joint fine-tuning experiment (not executed automatically)
bash my_work/experiments/exp_009_history_physics_adaptation/run.sh train \
  --run-name joint_seed0

# Resume an interrupted run, preserving learned modules, Critic, optimizers,
# sampler state, RNG state and update counters. Replay/environment are refilled,
# as upstream; resume is not bitwise continuation of simulator trajectories.
bash my_work/experiments/exp_009_history_physics_adaptation/run.sh train \
  --resume my_work/outputs/exp_009_history_physics_adaptation/joint_seed0/ckpt_0000250.pt

# After actual fine-tuning: full existing LAFAN1 clips, all four fixed μ values,
# base vs adapted vs residual-disabled (no extra training)
OMP_NUM_THREADS=2 .venv/bin/python -u \
  my_work/experiments/exp_009_history_physics_adaptation/evaluate.py \
  --checkpoint my_work/outputs/exp_009_history_physics_adaptation/joint_seed0/ckpt_0005000.pt \
  --run-name lafan1_joint5000

# Optional --ood adds μ=0.2 as a separately marked stress test.
# --clips walk1_subject1 ... supports a clearly selected evaluation subset.

# Training figures and evaluation figures, only where measurements exist
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
  my_work/experiments/exp_009_history_physics_adaptation/report.py \
  --run-dir my_work/outputs/exp_009_history_physics_adaptation/joint_seed0 \
  --eval-dir my_work/outputs/exp_009_history_physics_adaptation/evaluation/lafan1_joint5000
```

`config.json` controls 5,000 iterations, checkpoint interval 250, residual scale,
loss weights and optional training-only early stopping. Early stopping is off by
default; when enabled it monitors original imagined objective, with specified
minimum iterations, patience and min_delta, never LAFAN1. `--max-hours` is an
optional wall-clock stop. No automatic second fine-tuning experiment is started.

Fresh run/evaluation directories and numbered checkpoints refuse overwrites.
`latest.json` is an owned mutable pointer; numbered `.pt` snapshots are immutable.
SIGTERM and Ctrl-C stop after an iteration. Nonfinite/OOM errors preserve an explicitly
named diagnostic snapshot when possible; partial-iteration snapshots are refused
for continuation, so resume the last completed numbered checkpoint. OOM reports measured
allocated/reserved peaks and retains the original 2048/32/4096 sizes.

## Evaluation and numerical validity

The multi-checkpoint evaluation is implemented in `evaluate_sweep.py` and
`sweep_report.py`: fixed 5k/10k/20k/80,862 checkpoints, all existing full LAFAN1
clips, μ=1.0/0.7/0.5/0.3, shared original and disabled-residual controls. Each
condition is saved atomically and reused on restart after input/protocol hashes
are checked. Reports are updated after each friction coefficient. No new training
or checkpoint selection is performed.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -u \
  my_work/experiments/exp_009_history_physics_adaptation/evaluate_sweep.py \
  --run-name sweep_5k_10k_20k_80862 --share-gpu
```

`--share-gpu` explicitly permits immediate GPU sharing; without it the worker
waits for other compute processes to finish. The active detached worker must not
be launched twice (a file lock prevents duplicate evaluation). Progress and phase
logs are `evaluation/sweep_5k_10k_20k_80862/status.json` and
`evaluation_sweep.log`; final/partial Chinese reports are in the sweep's
`analysis/RESULTS_ZH.md`. Partial reports clearly identify missing conditions.

`evaluate.py` uses existing checkpoint loader, env config, ClipResult and result
writer, with a local one-clip-per-world loop because community `run_benchmark`
does not accept friction callbacks. It preserves complete clips, start-frame
initialization, deterministic actions, disabled noise/other DR, original failure
terminations and full-length denominator. Evaluation parallelism is 64, independent
of the required 4096 **training** worlds. All policies use identical clip order,
seeds and physics; initial-state hashes are checked.

The live startup foot friction and every sampled active foot–ground contact are
verified, including constraint addresses. Only foot tangential coefficients
change, matching training ξ semantics. Other body contacts retain original values.
Standard evaluation physics remains 35 contacts /250 constraints /10 solver /
20 line-search iterations; numerical flags are recorded separately, not silently
resolved by changing the standard benchmark. Capacity-affected results require
paired numerical-control validation before being interpreted as robustness.

Residual-disabled action equality is checked **exactly** against the original
base on every identical observation. GPU contact simulation can still have small
numerical trajectory differences on repeated runs; exact policy equality does
not imply bitwise MuJoCo trajectory equality. Smoke-only checkpoints are refused
as fine-tuned evaluation models. Slip velocity remains unavailable until a
contact-point estimator is independently validated; no slip measurements are
invented. Body-position and joint-RMSE diagnostics are pre-action and conditional
on surviving states, while normalized return penalizes premature termination.

## Files and outputs

- `common.py`, `config.json`: immutable-source audit, bounded output IO, configuration.
- `models.py`: Encoder, Adapter and frozen-base wrapper.
- `joint.py`: local step/actor orchestration around community imagination/GAE.
- `train.py`, `run.sh`: inspect, validation, manual joint run and resume.
- `validate.py`, `test_models.py`: real-model gradient/full-size checks and CPU tests.
- `audit.py`: CPU checkpoint, optimizer and resume-restore checks, source hash audit.
- `eval_smoke.py`: real evaluator interface check, not a fine-tuned benchmark.
- `memory_monitor.py`: read-only process-memory measurements.
- `evaluate.py`: LAFAN1 comparisons and contact verification.
- `report.py`: English PNG/PDF plots and concise status report.

All output is under `my_work/outputs/exp_009_history_physics_adaptation/`, including
compiled caches, logs, TensorBoard, JSON/CSV, checkpoints, predictions and reports.
`validation/smoke_only.pt` is one diagnostic joint update, **not** a fine-tuned model.
Validation has no `metrics.csv` and does not produce fake 5,000-iteration curves.
The report manifest explicitly lists missing training/evaluation figures. Seven
requested figure types are implemented for future real logs/evaluation; only
available diagnostic measurements are plotted now.

Interpret improvements cautiously: an Adapter can exploit state/motion cues even
if μ̂ is inaccurate. Better tracking does not alone prove friction identification
or beneficial use of history. Additional inference diagnostics/history ablations
would be needed for mechanism claims, and are not additional training runs here.
# Final checkpoint versus community: expanded friction evaluation

Only `base` and `step_80862` are compared in run `final_vs_community_friction10`.
Friction points: 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0.
The training interval is [0.3, 1.2]; points outside it test out-of-range
robustness. New values inside a continuously randomized interval are not
evidence of out-of-distribution generalization. Encoder predictions are bounded
to [0.3, 1.2], so cannot match true friction outside that interval.

Run on the intended GPU in tmux (replace GPU index as needed):

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -u my_work/experiments/exp_009_history_physics_adaptation/evaluate_sweep.py --run-name final_vs_community_friction10 --final-only --frictions 1.0 0.7 0.5 0.3 0.4 0.2 0.15 1.2 1.5 2.0 --reuse-from sweep_5k_10k_20k_80862 --share-gpu >> my_work/outputs/exp_009_history_physics_adaptation/final_vs_community_friction10.log 2>&1
```

Compatible completed groups are copied with source hashes; originals stay
unchanged. Input, checkpoint, motion and evaluator hashes are checked before
reuse. Completed groups are skipped on restart; interrupted groups restart.
Forty full LAFAN1 clips per condition, 20 groups / 800 episodes total including
reused groups. No intermediate checkpoints or disabled residual runs are added.
The absence of new numerical repeats limits claims about small improvements.
Outputs, English PNG/PDF plots and CSV/JSON statistics live under
`my_work/outputs/exp_009_history_physics_adaptation/evaluation/final_vs_community_friction10/`.

# Random friction changes: original versus final checkpoint

Implementation: `evaluate_random_friction.py`, `random_friction_protocol.py`,
`random_friction_report.py`, `run_random_friction.sh`, `test_random_friction.py`.
No policy/world-model training or community source changes.

Protocol: first 3 seconds at μ=1.0, then independent Uniform(0.2,1.5) samples
every 3 seconds, applied **before** control steps 150, 300, 450, etc. Sequences
are deterministic by motion name and sequence seed, independent of model,
motion selection order and episode survival. Seeds 0, 1, 2 are sequence seeds;
environment seed remains 0. Each policy has one μ=1.0 no-switch control.
Forty full LAFAN1 motions imply 240 random episodes + 80 controls = 320 episodes.
These are 40 independent motion groups, not 320 independent motions.

From the repository root, run in tmux on the intended machine/GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash my_work/experiments/exp_009_history_physics_adaptation/run_random_friction.sh --run-name random_friction_final_vs_community > my_work/outputs/exp_009_history_physics_adaptation/random_friction.log 2>&1
```

Monitor:

```bash
tail -f my_work/outputs/exp_009_history_physics_adaptation/random_friction.log
cat my_work/outputs/exp_009_history_physics_adaptation/evaluation/random_friction_final_vs_community/status.json
```

The same command resumes completed groups (one condition × all selected clips).
Interrupted groups restart; NPZ is atomically saved before JSON commits the group.
A lock prevents duplicate workers. Changed checkpoint/data/protocol/source hashes
require a new run name. GPU index is local to the machine; this runner starts
immediately on the visible GPU. Do not start alongside another large evaluation.

Pilot (censored, never treat as full-motion performance):

```bash
CUDA_VISIBLE_DEVICES=0 bash my_work/experiments/exp_009_history_physics_adaptation/run_random_friction.sh --run-name random_friction_pilot_custom --clips walk1_subject1 walk1_subject2 --sequence-seeds 0 --max-seconds 12
```

CPU protocol tests:

```bash
.venv/bin/python -m pytest -q my_work/experiments/exp_009_history_physics_adaptation/test_random_friction.py
```

Outputs under `evaluation/<run-name>/`:

- `manifest.json`, `status.json`: protocol, hashes and progress.
- `groups/*.npz`: full-rate (50 Hz) actions, observations split into joint velocity,
  base angular velocity and projected gravity, root errors, estimated/actual μ,
  rewards, time, contacts and numerical flags. Full history/goal/state/action/reward
  retained for the initial 3-second pairing prefix. Complete planned schedules
  retained even when an episode fails. `active` masks inactive worlds; episode
  rows specify valid trace lengths. No reliable foot-contact slip estimator yet.
- `groups/*.json`: per-motion outcomes, actual switch times, geometry metadata,
  contact evidence and NPZ checksums. In-place writes retain Warp buffer pointers;
  only the 14 foot geoms' sliding friction changes. Plane and non-foot friction,
  other friction axes, noise and other physical parameters remain fixed.
- `analysis/episodes.csv`, `summary.csv`, `paired_effects.csv`,
  `pairing_audit.csv`, `switch_windows.csv`, `results.json`, `RESULTS_ZH.md`.
- English PNG/PDF: performance, switch-aligned excess tracking error, and an
  illustrative encoder/control trace. Plot labels identify censored pilots.

The actual solver contact friction is checked **every step**, per contact
`worldid`, for all active foot–plane constraints at the last physics substep.
No active contact means no contact-based verification at that instant, rather
than an assertion that airborne friction was verified. Both tangential contact
axes must agree with the requested μ. Buffer identity and unrelated model
parameters are checked. Switch times include ideal control time and actual
float32-accumulated simulator time; long float32 clock drift is accounted for
without changing the simulation clock. There is no reset/history reset/checkpoint
reload at switches; automatic termination resets only affect exhausted worlds,
which are excluded from that episode's measurements.

Strict pairing: within each policy, compare its random run against its own
no-switch control before 3 seconds; initial SHA must match and max absolute
state/history/goal/action/reward difference must be ≤1e-4. Different policies
are expected to produce different actions; cross-policy checks compare initial
state and schedule, not equality of pre-switch trajectories. Preliminary pilots
showed some prefix gates failing due to repeat divergence, so all failures are
retained in the audit. Such episodes stay in overall robustness statistics but
are excluded from strictly paired switch-response analysis; do not loosen the
gate after observing which policy wins.

Analysis: full-motion return denominator includes the planned full clip length,
even after early failure. Report both raw final-minus-original return and
`(final random − final control) − (original random − original control)`.
Bootstrap first averages sequence seeds within motion, then resamples motions.
Switch response uses complete common-survival 1-second windows and excludes
capacity-affected or prefix-unpaired runs. These windows have survival bias and
must not replace failure statistics. No arbitrary recovery time is reported.
Encoder predictions are bounded to [0.3,1.2], so estimates cannot match out-of-range
true values. Behavior/estimation changes alone do not establish online adaptation
or a causal benefit of history; fixed-encoder/history ablations remain future work.
