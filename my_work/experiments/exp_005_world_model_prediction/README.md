# Experiment 005：World Model Prediction Analysis

目标：用相同起点、实际执行动作和仿真上下文，比较社区 Learned World Model 与 MuJoCo 的预测状态。不训练，不重跑完整 Benchmark，不修改社区源码。

**已实际完成采集、World Model 推理和英文 PNG/PDF 图表生成；7 项 CPU 测试通过。**

## 1. 可行性结论

- `realmimic_g1_86k.pt` 包含完整 `wm` 权重和归一化统计，社区 `load_world_model()` 可严格加载。
- 推理接口：`WorldModel.step(state[71], action[29], xi[33]) → next_state[71]`，每步 0.02 秒。
- 状态为 `[qpos(36) | qvel(35)]`：去环境原点的世界位置、wxyz 四元数、关节位置、世界系线速度、机体系角速度、关节速度。网络使用 body-frame 特征，输出积分回相同状态表示。
- 模型不预测完整 MuJoCo 内部状态，例如接触缓存、求解器 warm-start 和执行器内部状态。
- 实验 004 缺少 `qvel`、执行动作和 `xi`，不能直接拿来计算预测误差。这里只复用其动作选择、社区环境和对照记录，重新采集必要的短诊断轨迹，不用差分伪造速度。

详细检查记录：`my_work/outputs/exp_005_world_model_prediction/feasibility.json`。

## 2. 运行检查和测试

从项目根目录执行：

```bash
bash my_work/experiments/exp_005_world_model_prediction/run.sh check
.venv/bin/python -m unittest discover \
  -s my_work/experiments/exp_005_world_model_prediction -p 'test_*.py'
```

`check` 在 CPU 加载预训练 World Model，不创建仿真。

## 3. 采集诊断轨迹（只运行一次）

```bash
bash my_work/experiments/exp_005_world_model_prediction/run.sh capture --device cuda:0
```

三个动作：`walk1_subject1`、`dance2_subject4`、`fallAndGetUp1_subject1`，分别对应正常行走和两个挑战动作。

每个动作从参考开头采集最多 512 步（10.24 秒），在首次失败／结束时停止。保持社区无噪声、无随机化、相同增益和 checkpoint 环境设置；仅关闭自动 reset，以保留真实终止状态和动作。

逐步读取 `StateIO`、`SimContext` 和 `joint_pos.executed_raw`；保存实际状态、执行动作及容量溢出标记。`executed_raw` 是归一化 PD 位置指令，不是力矩。关闭噪声后会验证它与 Policy 输出一致。

已有轨迹时脚本拒绝覆盖。不要为了生成图表再次采集。

## 4. 运行预测与画图（可重复）

```bash
bash my_work/experiments/exp_005_world_model_prediction/run.sh analyze --device cuda:0
```

仅使用已保存轨迹做预训练模型推理，不启动 MuJoCo。也支持 `--device cpu`。

- **Teacher-forced one-step**：每一步都从真实 `s[t]` 预测 `s[t+1]`。
- **Autoregressive rollout**：只在起点使用真实状态，之后递归预测；使用同一段已记录的动作及 `xi`，不让 Policy 在预测状态上重新选动作。
- 预测长度：1、4、8、16、32 步，即 0.02、0.08、0.16、0.32、0.64 秒。
- 各长度使用相同起始窗口；主曲线排除整个 32 步窗口内发生过溢出的样本。包含溢出的结果也保留在 CSV 中。

## 5. 查看输出

```text
my_work/outputs/exp_005_world_model_prediction/
├── feasibility.json
├── REPORT.md
├── metrics.csv
├── summary.json
├── 01_rollout_errors.png / .pdf
├── 02_one_step_and_rollout.png / .pdf
└── <motion>/
    ├── capture.json
    └── raw/                         # Git 忽略
        ├── transitions.npz          # 状态、执行动作、xi、终止及溢出标记
        └── prediction_errors.npz
```

`capture.json` 记录本次状态与实验 004 重叠部分的差异；`summary.json` 记录软件版本和代码哈希。英文实验报告由真实输出生成。

## 指标与边界

主指标直接复用社区 `StateSpec.error_vector`：70 维归一化状态误差的均方。另报告 Root 位置 RMSE（m）、Root 最短旋转角 RMSE（rad）、29 关节角 RMSE（rad）及速度 RMSE。归一化尺度来自 checkpoint，不用参考动作跟踪得分代替模型预测误差。

三个动作、每个一次短采集仅用于案例分析。舞蹈只覆盖开头约 10 秒，不包含实验 004 约 60 秒处的失败。完整序列失败不表示每个短窗口都难预测；误差增长也不能直接解释 Policy 失败原因。

## 本次实际结果

| 动作 | 采集步数 | TF one-step 归一化 MSE | AR32 归一化 MSE | AR32 Root 位置 RMSE | AR32 关节角 RMSE |
|---|---:|---:|---:|---:|---:|
| walk1_subject1 | 512 | 0.000207886 | 0.00649816 | 0.0138697 m | 0.00989325 rad |
| dance2_subject4 | 512 | 0.000272961 | 0.0140472 | 0.0223159 m | 0.0145245 rad |
| fallAndGetUp1_subject1 | 399（失败终止） | 0.0203753 | 0.291615 | 0.0552975 m | 0.0642523 rad |

三段轨迹均无容量溢出。AR 曲线分别使用 61、61、46 个公共起点窗口；TF one-step 使用各自全部有效转移，二者样本集不同。跌倒起身误差更高；舞蹈开头的预测较好，不能据此解释完整序列后期失败。

与实验 004 重叠部分的最大逐分量 qpos 差异分别为约 2.79e-6、2.99e-5、0.0464；qpos 混合米、四元数分量和弧度，此数仅是复现诊断，不是统一物理误差。差异已保留，WM/MuJoCo 比较统一使用本次新记录的真实状态和输入。
