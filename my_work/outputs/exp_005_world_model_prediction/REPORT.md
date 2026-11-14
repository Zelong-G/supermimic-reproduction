# Experiment 005: World Model Prediction Analysis

Frozen pretrained World Model; no training. New short diagnostic transitions are required because Experiment 004 did not log velocities/actions/context.

## Protocol

Teacher forcing: predict each real s[t+1] from real s[t], executed a[t] and measured xi[t]. Autoregression: initialize at real s[t], then recursively apply the SAME logged actions and xi for h steps; the policy is not rerun in predicted states.

Primary autoregressive curves use shared start indices for horizons 1,4,8,16,32. Windows containing any capacity overflow in the full 32 steps are excluded for every horizon. Inclusive metrics and one-step-only overflow filters remain in metrics.csv.

## Actual captures

| Motion | Transitions | Stop | Overflow transitions | Max qpos difference vs Exp004 overlap |
|---|---:|---|---:|---:|
| walk1_subject1 | 512 | diagnostic_cap | 0 | 2.78652e-06 |
| dance2_subject4 | 512 | diagnostic_cap | 0 | 2.99457e-05 |
| fallAndGetUp1_subject1 | 399 | termination | 0 | 0.0464231 |

## Prediction errors

One-step rows use all overflow-free transitions; AR32 uses shared 32-step windows. Their sample counts differ.

| Motion | TF1 normalized MSE | AR32 normalized MSE | AR32 root pos RMSE (m) | AR32 orientation RMSE (rad) | AR32 joint RMSE (rad) | Clean AR windows |
|---|---:|---:|---:|---:|---:|---:|
| walk1_subject1 | 0.000207886 | 0.00649816 | 0.0138697 | 0.0215938 | 0.00989325 | 61 |
| dance2_subject4 | 0.000272961 | 0.0140472 | 0.0223159 | 0.0308152 | 0.0145245 | 61 |
| fallAndGetUp1_subject1 | 0.0203753 | 0.291615 | 0.0552975 | 0.212921 | 0.0642523 | 46 |

## Metric definitions

- State is the community 71-D origin-free [qpos|qvel], root quaternion wxyz, root linear velocity in world coordinates and angular velocity in body coordinates. No yaw/position realignment is performed.
- Normalized state MSE is the community StateSpec.error_vector squared and averaged over 70 dimensions, then over samples; quaternion error is a rotation vector. Checkpoint scales: pos 0.05 m, rotation 0.1 rad, linear velocity 0.5 m/s, angular velocity 1 rad/s, joint pos 0.1 rad, joint velocity 2 rad/s.
- Root position/orientation RMSE use vector squared norms averaged over windows, then square root. Joint RMSE averages squared errors over windows and all 29 joints. Velocity metrics are in metrics.csv. Body angular velocities are compared in the community state convention.
- The WM uses body-frame features, integrates back to the same world/root/joint state representation and advances 0.02 s per step. xi contains measured friction, encoder biases and torso COM offsets.
- Action is executed_raw (29 normalized PD position commands), not torque or processed joint-target radians. Noise is disabled; saved clean and executed actions are checked for equality.

## Limits

- Only three selected clips, one deterministic run each; this is not a population benchmark or uncertainty estimate.
- At most 512 control steps (10.24 s) per case. Dance uses its early section, not the approximately 60-second failure seen in Experiment 004. Motion labels do not imply every window is difficult.
- Terminated capture stops immediately and includes the true terminal state (auto_reset=False); windows do not cross resets or continue after failure.
- The generalized state omits solver/contact caches. This tests prediction along logged MuJoCo transitions, not exact simulator cloning or a reinitialized state-write counterfactual.
- Growth of multi-step error is a descriptive result, not proof of any single cause of policy failure. Distinguish model error, action sensitivity and numerical/contact effects.
- These are real observations; nonmonotonic error curves and differences from Experiment 004 are retained.

## Reproduce

From repository root:

```bash
bash my_work/experiments/exp_005_world_model_prediction/run.sh analyze --device cuda:0
```

Analyze reuses saved transitions and does not create a MuJoCo environment. See the experiment README for initial capture and feasibility commands.
