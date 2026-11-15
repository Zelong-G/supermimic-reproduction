# Experiment 008 — Friction Adaptation on Unseen LAFAN1 Motions

Frozen `weights/realmimic_g1_86k.pt`; existing LAFAN1 only. No downloads,
BONES-SEED, policy/world-model training, or modifications to community code / Exp001–007.
All outputs: `my_work/outputs/exp_008_friction_adaptation/`.

## Run from repository root

```bash
# Real contact inspection (small GPU job)
OMP_NUM_THREADS=2 .venv/bin/python -u my_work/experiments/exp_008_friction_adaptation/probe.py
# config.json is supplied; prepare can reproduce selection and turn_candidates.json
bash my_work/experiments/exp_008_friction_adaptation/run.sh pilot
bash my_work/experiments/exp_008_friction_adaptation/run.sh pilot_high
bash my_work/experiments/exp_008_friction_adaptation/run.sh pilot_solver
bash my_work/experiments/exp_008_friction_adaptation/run.sh pilot_precision
# Run sequentially, after inspecting pilot contact validity and numerical flags
bash my_work/experiments/exp_008_friction_adaptation/run.sh fixed
bash my_work/experiments/exp_008_friction_adaptation/run.sh repair
bash my_work/experiments/exp_008_friction_adaptation/run.sh switch
bash my_work/experiments/exp_008_friction_adaptation/run.sh analyze
OMP_NUM_THREADS=2 .venv/bin/python my_work/experiments/exp_008_friction_adaptation/verify.py
```

Capture refuses to overwrite an existing episode. Analysis reuses raw data.
Move the complete experiment output directory to an archive before recapture;
do not mix old and new episodes. Never run these GPU commands concurrently.

## Protocol

12 preselected existing 500-frame segments, four descriptive strata with three
segments each, five fixed friction values (60 episodes). Seed 0, deterministic
policy, all observation/action/reset noise and other DR disabled. Repeated
zero-noise seeds are not independent samples. The existing Exp003 segment
evaluation determines selection and historical labels. Turning is a walking
sub-stratum, selected by pelvis unwrapped yaw range >=45 degrees, with distinct
parent motions and historical success / no overflow. It is not an invented
LAFAN1 filename/category. Sources can overlap between walking and turning.

Part B uses the three walking segments with paired control and two switched
conditions (9 episodes +3 numerical control repeats), only after confirming fixed/control success. At step
150 (3.00 s), write foot tangential friction in-place; keep robot, checkpoint,
reference phase and observation history. Verify pre-switch state/history/goal/
action/reward within 1e-4 absolute tolerance and actual contact friction after switching.
This tolerance covers ~5e-5 floating-point velocity differences in pilot control
prefixes; separate unchanged-control repeats quantify the numerical floor. Repeats
are technical checks, not additional independent statistical samples.
The final pairing gate accepted `walk1_subject1__seg000` (3 primary episodes +1
control repeat). The other two attempted clips exceeded the strict prefix
tolerance in a control repeat or intervention and are retained in
`pairing_audit.json` with all raw data, excluded from online-response conclusions.
This is a numerical reproducibility limitation. Main switch figures use only the
accepted clip; the capture count remains 12 attempted episodes.

Recovery is declared before analysis: after a detected violation, first start of
50 consecutive steps with paired excess pelvis position error <=0.05 m AND
orientation error <=0.10 rad. Time is measured from switch; no qualifying window
is `not_recovered` with null time. No violation is `no_detected_disruption`, not
zero-time recovery. Observation ends at first failure or segment end.

Part C fits independent ridge probes (fixed alpha 100), comparing 93-D current
observation and 930-D ten-frame history. Group all segments/conditions from a
parent motion in the same one of three held-out folds; standardize on training
data only. Use common surviving time prefix across μ per segment, start at 0.5 s,
stride 5. These probes do not update the policy or establish causal use of history.

## Source and physics audit

- `src/supermimic/policy/nets.py`: actual checkpoint selects running normalization,
  ELU, ten-frame proprioception (930) and SONIC/pelvis_ori10 goal (650). Policy has
  no ξ argument. Goal includes reference orientation relative to current pelvis.
- `src/supermimic/envs/tracking_env_cfg.py`: 0.005 s ×4 =0.02 s control step;
  history oldest/newest separation 0.18 s (ten sampling intervals 0.20 s).
  Training foot friction reset DR U[0.3,1.2], absolute shared axis 0; evaluation
  startup foot friction=1.0. Local wrapper disables all unrelated randomization.
- `scripts/eval_benchmark.py`, `src/supermimic/eval/benchmark.py`: reuse checkpoint
  loader and env_kwargs; replicate full-clip reward denominator and first failure.
  Local runner sets auto_reset=False to preserve terminal outcomes.
- Exp003: historical source/segment results; Exp004: reference-aligned root pose
  errors and start sampling; Exp005: StateIO and actual action/observation capture.
  Their existing files are read only. Exp004 lacks complete diagnostic observations;
  Exp005 has only one friction, so fresh paired captures are necessary.
- Installed mjlab `envs/mdp/dr/geom.py`: default axis 0 only. Installed Warp
  `_src/collision_core.py:contact_material_params`: priority determines friction;
  equal-priority geoms use componentwise maximum. In actual model feet priority 1,
  terrain priority 0, condim=3, no explicit pairs. Ground remains μ=1.0. Only the
  14 foot geoms change. This matches training foot-friction semantics, not uniform
  whole-floor material changes; non-foot contacts retain original friction.
- In-place writes to already-expanded Warp arrays retain CUDA graph pointers.
  Inspect real `wp_data.contact.friction`, geom IDs and `efc_address`; verify every
  active sampled foot-ground contact. Full contact evidence is in episode JSON;
  min/max friction and active contact counts are in every NPZ. Counts are last
  physics-substep samples; no claim of reliable contact-point sliding velocity.
- [MuJoCo contact parameter rules](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters).
- `overflow_mask` distinguishes lost-capacity flags from ITERATIONS (512) /
  LS_ITERATIONS (1024). Pilot_high expands buffers; pilot_solver raises line-search
  limit to inspect numerical sensitivity. Formal settings and flags are saved.

Formal captures use iterations=100, ls_iterations=80 rather than community
defaults 10/20, fixed equally across μ. The successful small precision pilot and
default-setting comparisons are retained. Initial formal buffers are 35/250.
`repair` identifies any source segment with a capacity flag and reruns ALL its
five μ conditions with nconmax=140 / njmax=1000; original sets remain in
`capacity_affected_initial`. This only changes storage/numerics, not contact
materials or physical parameters. Per-case paired settings are identical;
`fixed/repair_metadata.json` lists replacements. Part B uses 140/1000 throughout.
Earlier incomplete exploratory settings remain in `exploratory_iter10` and
are excluded from formal summaries. Numerical flags are reported separately;
capacity-affected data must pass repair before interpretation.

## Outputs and limits

`contact_probe.json`, `config.json` (code directory), per-stage metadata, each
episode NPZ/JSON, `episodes.csv/json`, `fixed_summary.csv/json`,
`switch_summary.csv/json`, `switch_timeseries.csv`, `history_inference.csv/json`,
`history_predictions.csv`, `validation.json`, `analysis_provenance.json`,
`REPORT.md`, English PNG/PDF figures. Raw NPZ contains state, reference,
observations, goal, actions, reward, errors, contact parameters/counts, termination,
and numerical flags. Actions are normalized PD commands, not torque.

Return=sum reward/(0.1×500), including failure penalty through full duration.
Root errors are pre-action pelvis Euclidean distance / geodesic orientation,
averaged over surviving states. Time-to-failure is null for censored completions.
Per-stratum n=3; report counts rather than treating them as population estimates.
Score intervals resample source motions; overall failure Wilson intervals ignore
source overlap and are explicitly descriptive.

Distinguish robustness, online feedback response, implicit inference, and
out-of-range friction generalization. Action changes alone do not prove friction
identification. Probe performance does not prove history improves control. A
history ablation would be required for causal attribution. Failure under low μ
does not establish a missing-ξ defect. No new network is proposed. Ten-second
segment outcomes do not establish long-clip success; exact membership of LAFAN1
in community pretraining cannot be independently certified by the checkpoint.
