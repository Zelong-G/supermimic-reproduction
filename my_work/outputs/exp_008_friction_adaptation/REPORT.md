# Experiment 008: Friction Adaptation on Unseen LAFAN1 Motions

Frozen community checkpoint; selected existing 10-second LAFAN1 segments. No BONES-SEED, downloads, policy training or world-model training. "Unseen" follows the prior LAFAN1 evaluation protocol; the exact pretraining corpus membership cannot independently be proven from the checkpoint.

## Fixed friction results

| μ | Return | Failures / episodes | Root position error (m) | Root orientation error (rad) | Valid steps | Capacity flags | Solver-limit flags |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.8778 | 2/12 | 0.1272 | 0.1462 | 5892 | 0 | 0 |
| 0.7 | 0.8828 | 3/12 | 0.1105 | 0.1289 | 5870 | 0 | 0 |
| 0.5 | 0.8779 | 2/12 | 0.1075 | 0.1360 | 5877 | 0 | 0 |
| 0.3 | 0.8600 | 2/12 | 0.1715 | 0.1316 | 5874 | 0 | 0 |
| 0.15 | 0.7436 | 2/12 | 0.4751 | 0.2190 | 5945 | 0 | 0 |

Per-category n=3 episodes per μ; see fixed_summary.csv for failure counts and uncertainty. One deterministic seed; rerunning identical zero-noise seeds would not add independent evidence. Selected sources overlap between walking and turning; overall bootstrap clusters by source. Wilson failure intervals are descriptive and do not account for this overlap.

## Actual motions

| Stratum | Existing segment | Parent motion | Reference yaw range (deg) | Historical score |
|---|---|---|---:|---:|
| Walking | walk1_subject1__seg000 | walk1_subject1 | 18.6 | 0.9802 |
| Walking | walk1_subject2__seg000 | walk1_subject2 | 117.4 | 0.9802 |
| Walking | walk1_subject5__seg000 | walk1_subject5 | 367.3 | 0.9620 |
| Dancing | dance1_subject1__seg000 | dance1_subject1 | 34.6 | 0.9592 |
| Dancing | dance1_subject3__seg000 | dance1_subject3 | 9.2 | 0.9871 |
| Dancing | dance2_subject4__seg005 | dance2_subject4 | 386.3 | 0.6857 |
| Fall-and-Get-Up | fallAndGetUp1_subject1__seg000 | fallAndGetUp1_subject1 | 146.9 | 0.6110 |
| Fall-and-Get-Up | fallAndGetUp2_subject2__seg000 | fallAndGetUp2_subject2 | 71.7 | 0.8107 |
| Fall-and-Get-Up | fallAndGetUp3_subject1__seg000 | fallAndGetUp3_subject1 | 125.9 | 0.8832 |
| Turning | walk1_subject5__seg015 | walk1_subject5 | 1087.1 | 0.8657 |
| Turning | walk3_subject5__seg003 | walk3_subject5 | 844.3 | 0.9098 |
| Turning | walk1_subject2__seg024 | walk1_subject2 | 615.7 | 0.8931 |

Turning is a quantitative walking sub-stratum (pelvis unwrapped yaw range >=45 degrees), not a new LAFAN1 action label. Turning examples are the largest yaw ranges among historically successful, overflow-free segments, with distinct parents. Other strata use preselected source clips from Exp003, not exhaustive or randomly sampled categories. Walking can also include turns.

## Sudden change

Attempted 3 clips / 12 episodes including numerical repeats. Eligible clips: ['walk1_subject1__seg000']. Excluded clips: ['walk1_subject2__seg000', 'walk1_subject5__seg000']; their control-repeat or switched prefixes exceeded the predeclared 1e-4 absolute tolerance before friction changed. Full data and per-term differences remain in pairing_audit.json. Main switch figures/statistics use eligible clips only; the exclusion is a numerical reproducibility limitation, not evidence of policy friction failure.

Switch writes the existing Warp buffer at step 150, before integrating [3.00, 3.02) s. No episode reset, history reset, robot reinitialization, or checkpoint reload occurs at the switch. All prefixes are checked for state, history, goal, action and reward agreement within 1e-4 absolute tolerance, covering small floating-point velocity differences (~5e-5) in pilot controls. Three extra unchanged-control repeats measure the numerical floor; they are not independent behavioral samples. Actual simulator times and buffer pointers are checked. Signals plotted at pre-action timestamps; reward and contacts are from the ensuing control interval. Contacts are sampled at the last physics substep only.

Recovery criterion, declared before analysis: First post-switch start of 50 consecutive steps with paired excess root position error <=0.05 m and excess orientation error <=0.10 rad; meaningful only after a prior violation, otherwise no detected disruption. Right-censored at failure or segment end.

| Clip | Target μ | Prefix max difference | Mean excess position (m) | Mean excess orientation (rad) | Failed | Recovery | Time (s) |
|---|---:|---:|---:|---:|---|---|---|
| walk1_subject1__seg000 | 1.0 | 0 | 0.0000 | 0.0000 | False | no_detected_disruption | None |
| walk1_subject1__seg000 | 0.3 | 4.66e-05 | 0.0130 | 0.0001 | False | not_recovered | None |
| walk1_subject1__seg000 | 0.15 | 3.35e-05 | 0.5334 | 0.0551 | False | not_recovered | None |

## History information probe

| Features | Episode-weighted MAE | RMSE | R² vs training mean |
|---|---:|---:|---:|
| Single frame | 0.2627 | 0.3145 | -0.1036 |
| 10-frame history | 0.2780 | 0.3402 | -0.2920 |
| Training-mean baseline | 0.2560 | 0.2993 | 0.0000 |

Exploratory diagnostic ridge regression only. Source-grouped folds prevent neighboring frames and conditions of the same source crossing train/test. All five μ values appear in training: this is not inference on unseen friction labels. The probe may exploit motion-dependent feedback responses and action correlations. It does not establish that the policy infers μ or uses its history beneficially. Small source count and common-prefix survivor selection limit generalization. Contact excitation is necessary; low-dynamic windows can be uninformative.

## Interface and contact audit

- Actual checkpoint policy: history=10, observation=930 (29 joint pos +29 joint vel +3 base angular vel +3 projected gravity +29 previous actions, each ten frames), goal=650 (10×59 SONIC reference features +60 pelvis_ori10 features), running normalization and ELU. No ξ input to policy; world model context is separate.
- Physics step 0.005 s, decimation 4, control step 0.02 s. Ten frames span 0.18 s from oldest to newest; 0.20 s when counted as ten sampling intervals. Reset history is backfilled.
- Training reset DR: absolute shared tangential foot friction U[0.3,1.2], 14 foot geoms. Evaluation startup fixes those geoms at 1.0. Ground μ=1.0, priority 0; feet priority 1, condim=3. No explicit contact pairs.
- Geom metadata stores compiled CPU defaults (feet initially 0.6); startup events set the live Warp feet to 1.0. Episode mu and contact friction are the live measurements, not the CPU defaults.
- Formal solver iterations=100, line-search iterations=80, uniformly across friction conditions (community defaults 10/20). Small pilots establish feasibility and numerical sensitivity; incomplete iter10 exploration is excluded. For any capacity-affected source segment, ALL five conditions are rerun with nconmax=140, njmax=1000; initial records remain in capacity_affected_initial. fixed/repair_metadata.json lists these paired replacements. Other segments retain 35/250; Part B uses 140/1000 throughout. These are storage/solver changes, not changed material parameters; validation must confirm zero capacity errors in final data.
- In-place writes to the already-expanded geom_friction array preserve CUDA graph references. Every active sampled foot-ground contact is checked against requested μ. contact_probe.json and per-episode contact_evidence retain full 5D contact friction plus constraint addresses.
- Only foot tangential friction changes; torsional/rolling coefficients and all other geoms stay fixed. This matches training ξ semantics. In fall/get-up, hands/torso ground contacts remain at their original friction; this experiment is not a uniform slippery-floor material change.
- MuJoCo priority rule: https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters ; installed Warp implementation: collision_core.contact_material_params.
- All noises and unrelated DR disabled, paired reference-frame initial states. Returns use full segment denominator 0.1×500, including early failure penalty. Errors are pelvis world-pose Euclidean/geodesic errors before action (not torso reward anchors), averaged over surviving states. Time to failure is null for censored completions.
- Raw states/actions/observations/velocities/contacts/rewards retained in NPZ; episode CSV/JSON and switch time-series CSV preserve statistics. Reliable contact-point sliding velocity was not implemented; contact count alone is not slip.

## Interpretation limits

Friction robustness measures performance across fixed conditions. Online feedback response after a change is compatible with adaptation but not proof of learned dynamics identification. Implicit inference concerns information in observations; a diagnostic probe does not prove its causal use by the controller. μ=0.15 is outside the configured friction range, but checkpoint training exposure and all other dimensions are not independently audited. Failure cannot be attributed to missing ξ input. Causal evidence for history improving control requires matched history ablation/control experiments, which are outside this first behavioral study. Ten-second segment success does not imply full-clip success. No network changes are motivated by these results alone.

## Figures

1. figures/01_tracking_score.png / .pdf
2. figures/02_failure_rate.png / .pdf
3. figures/03_sudden_change_errors.png / .pdf
4. figures/04_policy_observation_changes.png / .pdf
5. figures/05_history_friction_inference.png / .pdf
6. figures/06_fixed_root_errors.png / .pdf
