# Training Report

Run status: **completed**. Normal completion verified: **True**.

## Training stages

World Model-only updates through iteration 500; subsequent iterations update World Model, Policy and Critic. Initial data collection precedes these updates.

## Progress

- Target iterations: 86000.
- Verified final completed iterations: 86000.
- Verified final environment steps: 215040.
- Last logged iteration: 86000 (a lower bound after interruption).
- Total process wall time (seconds, including startup): 39493.672375139024.
- Latest checkpoint: ckpt_latest.pt; checkpoint iteration: 86000.

## Metric changes

| Metric | First finite value | Last finite value | Nonfinite samples |
|---|---:|---:|---:|
| wm/loss | 5.398882865905762 | 0.3128361701965332 | 0 |
| wm/state_loss | 5.384881019592285 | 0.3128216564655304 | 0 |
| wm/term_bce | 0.014001752249896526 | 1.4534565707435831e-05 | 0 |
| wm/online_nse_8step | 0.05567729100584984 | 0.0042941635474562645 | 0 |
| policy/imag_objective | 1.1428091526031494 | 9.554797172546387 | 0 |
| policy/critic_loss | 0.05397030711174011 | 0.00313295004889369 | 0 |

Loss, imagination objective and online prediction metric are different quantities. The online prediction metric is measured on training replay, not a held-out Benchmark.

## Logging and plotting

Trailing arithmetic mean of up to 5 logged points; reset at event file/step restarts and nonfinite values; no interpolation.
Raw logged samples, including nonfinite and duplicate/restart points, remain in metrics/scalars.csv. Existing console it/s is reused; TensorBoard time differences are a fallback. Five separate PNG/PDF curves are generated only for available data.

## Memory and stability

PyTorch counters exclude Warp; process peak is sampled every 30 s and may miss brief spikes.
GPU memory (MiB): {'gpu_torch_peak_allocated_mib': 16804.52197265625, 'gpu_torch_peak_reserved_mib': 17574.0, 'gpu_sampled_process_peak_mib': 20556}.

- Positive diagnostic recorded: sim/overflow_nefc
- Positive diagnostic recorded: sim/overflow_iterations
- Positive diagnostic recorded: sim/overflow_ls_iterations
- Positive diagnostic recorded: sampler/unphysical_episodes
- Positive diagnostic recorded: sampler/unphysical_joint_vel

## Provenance

training_summary.json includes complete configuration, seed, runtime versions, source hashes, commit and initialization/resume flags when available. A missing value means unmeasured, not zero.

Logs flush at 15-second intervals without adding evaluations. SIGKILL, power loss and failures before a log point can still lose unflushed or not-yet-emitted data; no checkpoint is created by this analyzer.
