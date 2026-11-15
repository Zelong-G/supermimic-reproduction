# Experiment 009 status report

Generated plots: ['validation_loss_and_gradients', 'validation_gpu_memory'].
Not generated (no measurements): ['01_joint_training_loss', '02_tracking_reward', '03_friction_mae', '07_residual_magnitude', '04_predicted_friction', '05_tracking_vs_friction', '06_failure_vs_friction'].

No claim of improved low-friction tracking or preserved standard-friction performance is possible until the manual joint fine-tuning and paired LAFAN1 evaluation have actually run. Validation-only scratch updates do not count as fine-tuning results.

Full-size validation status: passed.
Full update measurements: {'imag_objective': 9.483575820922852, 'friction_mse': 0.07187369465827942, 'friction_mae': 0.2325974851846695, 'action_regularization': 0.0, 'total_loss': -9.482856750488281, 'residual_abs_mean': 0.0, 'imag_return_undisc': 2.9539942741394043, 'imag_reward_mean': 0.09232540428638458, 'imag_survival_H': 0.9997200965881348, 'valid_time_weight': 65526.2421875, 'encoder_grad_norm': 0.0003360280825290829, 'adapter_grad_norm': 0.009524115361273289, 'joint_grad_norm': 0.009530042298138142, 'critic_loss': 0.003587572369724512, 'critic_value_mean': 9.545926094055176, 'critic_target_mean': 9.548080444335938, 'seconds': 261.88523624799564, 'peak_allocated_bytes': 15645487616, 'peak_reserved_bytes': 16382951424}.
Frozen hashes: {'policy': '08de103dff129041cc541511a219285945482eeb8c2876c4fe03126cdac92779', 'world_model': 'b5ec7b7b94cb62c7e8ee26f5067965dfc6c717ae8fbe9dd72b37f875467274fc'}.
