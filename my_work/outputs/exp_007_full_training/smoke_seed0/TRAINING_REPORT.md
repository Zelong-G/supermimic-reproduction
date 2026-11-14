# Training Report

Run status: **completed**. Normal completion verified: **False**.

## Training stages

World Model-only updates through iteration 4; subsequent iterations update World Model, Policy and Critic. Initial data collection precedes these updates.

## Progress

- Target iterations: 12.
- Verified final completed iterations: 12.
- Verified final environment steps: 70.
- Last logged iteration: 12 (a lower bound after interruption).
- Total process wall time (seconds, including startup): 51.4837325590197.
- Latest checkpoint: ckpt_latest.pt; checkpoint iteration: 12.

## Metric changes

| Metric | First finite value | Last finite value | Nonfinite samples |
|---|---:|---:|---:|
| wm/loss | 46.32046127319336 | 15.081737518310547 | 0 |
| wm/state_loss | 46.307518005371094 | 15.052934646606445 | 0 |
| wm/term_bce | 0.012943458743393421 | 0.028803354129195213 | 0 |
| wm/online_nse_8step | 0.7066022157669067 | 0.410597562789917 | 0 |
| policy/imag_objective | 0.5088358521461487 | 0.606469988822937 | 0 |
| policy/critic_loss | 0.06630443036556244 | 0.06798524409532547 | 0 |

Loss, imagination objective and online prediction metric are different quantities. The online prediction metric is measured on training replay, not a held-out Benchmark.

## Logging and plotting

Trailing arithmetic mean of up to 5 logged points; reset at event file/step restarts and nonfinite values; no interpolation.
Raw logged samples, including nonfinite and duplicate/restart points, remain in metrics/scalars.csv. Existing console it/s is reused; TensorBoard time differences are a fallback. Five separate PNG/PDF curves are generated only for available data.

## Memory and stability

PyTorch counters exclude Warp; process peak is sampled every 30 s and may miss brief spikes.
GPU memory (MiB): {'gpu_torch_peak_allocated_mib': 349.970703125, 'gpu_torch_peak_reserved_mib': 402.0, 'gpu_sampled_process_peak_mib': 956}.

- 42 nonfinite scalar samples; raw CSV preserves them.
- Positive diagnostic recorded: sim/overflow_iterations
- Positive diagnostic recorded: sim/overflow_ls_iterations

## Provenance

training_summary.json includes complete configuration, seed, runtime versions, source hashes, commit and initialization/resume flags when available. A missing value means unmeasured, not zero.

Logs flush at 15-second intervals without adding evaluations. SIGKILL, power loss and failures before a log point can still lose unflushed or not-yet-emitted data; no checkpoint is created by this analyzer.
