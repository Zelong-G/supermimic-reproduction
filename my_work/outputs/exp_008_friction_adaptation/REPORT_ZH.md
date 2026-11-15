# Experiment 008：研究发现与面试表述

已完成冻结社区权重的真实仿真、接触参数验证、固定摩擦试验、在线切换和独立历史诊断。未修改社区源码或 Experiment 001–007，未重新训练 Policy / World Model，未使用 BONES-SEED、下载数据或 git push。

## 实际设置

- checkpoint：`weights/realmimic_g1_86k.pt`。
- Policy 实际输入：930 维、10 帧本体观测历史，以及 650 维 SONIC / pelvis_ori10 参考目标；不直接接收 ξ。
- 控制间隔 0.02 s；10 帧首末间隔 0.18 s，按十个采样周期计为 0.20 s。
- 训练代码的摩擦随机化：14 个脚部 geom 切向摩擦，共享采样范围 [0.3, 1.2]；标准评测设为 1.0。
- 实验修改同一组脚部参数，地面和非脚部 geom 保持原值。脚部 priority=1 高于地面的 0；实际 Warp 接触摩擦与请求值相符，并具有有效约束地址。不是仅修改配置后推定生效。跌倒时手、躯干等接触不会变成统一低摩擦地面。
- 全部噪声和其他随机化关闭，seed=0；相同片段配对参考初始状态。重复无噪声 seed 不被当作独立样本。
- 正式求解器上限统一为 iterations=100 / ls_iterations=80，社区默认为 10/20。容量受限片段的全部五个条件均用 140/1000 接触/约束缓冲重新运行；原记录保留。最终固定试验 60 个 Episode 的容量和求解器上限标记均为零。
- `walk1_subject5__seg000` 在 μ=0.15 的小缓冲记录中失败，在扩大缓冲的正式配对记录中完成。这个数值混杂导致的失败没有归因于 Policy。

## 实际动作

全部来自现有 `data/npz/lafan1_seg10/`，每段计划 500 步、10 秒。

| 类型 | 文件 stem |
|---|---|
| Walking | walk1_subject1__seg000；walk1_subject2__seg000；walk1_subject5__seg000 |
| Turning | walk1_subject5__seg015；walk3_subject5__seg003；walk1_subject2__seg024 |
| Dancing | dance1_subject1__seg000；dance1_subject3__seg000；dance2_subject4__seg005 |
| Fall-and-Get-Up | fallAndGetUp1_subject1__seg000；fallAndGetUp2_subject2__seg000；fallAndGetUp3_subject1__seg000 |

Turning 是现有 walking 中按 pelvis 连续 yaw 范围≥45°选出的转向子集，不是虚构的文件或独立动作标签。其三个片段的 yaw 范围约 1087°、844°、616°。选段依据 Exp003 真实评测记录；该子集进一步要求历史成功且无溢出。Walking 与 Turning 的父运动有重叠：12 个片段来自 10 个父轨迹。

## Part A

| μ | 平均归一化回报 | 失败数 / Episode 数 | 平均 pelvis 位置误差 m | 平均姿态误差 rad | 实际执行步数 |
|---|---:|---:|---:|---:|---:|
| 1.0 | 0.8778 | 2/12 | 0.1272 | 0.1462 | 5892 |
| 0.7 | 0.8828 | 3/12 | 0.1105 | 0.1289 | 5870 |
| 0.5 | 0.8779 | 2/12 | 0.1075 | 0.1360 | 5877 |
| 0.3 | 0.8600 | 2/12 | 0.1715 | 0.1316 | 5874 |
| 0.15 | 0.7436 | 2/12 | 0.4751 | 0.2190 | 5945 |

回报分母统一为完整片段最大回报 `0.1×500`；提前失败不能因分母变短而获得虚高分。位置和姿态误差基于执行前 pelvis 状态，平均范围依赖存活时长；共同时间前缀的配对增量另见 `paired_fixed_deltas.csv`。完成者 time-to-failure 为 null。

每个类型、每个 μ 都是 3 个片段 / 3 个 Episode。失败数为：

| μ | Walking | Turning | Dancing | Fall-and-Get-Up |
|---|---:|---:|---:|---:|
| 1.0 | 0/3 | 0/3 | 0/3 | 2/3 |
| 0.7 | 0/3 | 0/3 | 1/3 | 2/3 |
| 0.5 | 0/3 | 0/3 | 1/3 | 1/3 |
| 0.3 | 0/3 | 0/3 | 1/3 | 1/3 |
| 0.15 | 0/3 | 0/3 | 1/3 | 1/3 |

训练范围内的总体表现相近，动作之间并非单调随摩擦变差。μ=0.15 相对 μ=1.0 的平均回报降低约 15.3%，配对回报增量为 -0.1343；父轨迹聚类 bootstrap 的描述性 95% 区间约 [-0.2295, -0.0412]。这是有意选取的小样本，不能作为整个 LAFAN1 的总体区间。

μ=0.15 的 Turning 平均位置误差约 1.280 m，但三个片段都未触发失败终止。标准终止检查主要关注参考高度和倾斜，成功标记会遗漏全局位置漂移。因此 failure rate 不变不代表跟踪性能不变。

## Part B

尝试三个 Walking 片段，各采集 control、1.0→0.3、1.0→0.15 和额外 unchanged-control repeat，共 12 个 Episode。严格的预设切换前绝对阈值为 1e-4；另两条片段在重复 control 或干预条件的速度 / 动作前缀中超过阈值，其全部数据和排除原因保留在 `pairing_audit.json`。主要结论只使用通过质量门限的 `walk1_subject1__seg000`，即 3 个主要 Episode +1 次技术重复。

该片段切换前状态/历史最大差异约 3.35e-5–4.66e-5；control repeat 为 5.17e-5。主要 root position 前缀差异仅约 1e-7 m。实际切换时间约 3.000021 s，切换后真实接触摩擦正确；参数缓冲指针保持不变，没有重置历史、机器人或参考，也没有重载 checkpoint。

| 条件 | 切换后相对 control 的平均位置误差增量 m | 最大增量 m | 平均姿态误差增量 rad | 失败 | 恢复事件 |
|---|---:|---:|---:|---|---|
| 1.0→0.3 | 0.0130 | 0.0728 | 0.00014 | 否 | 未达到预设恢复判据 |
| 1.0→0.15 | 0.5334 | 1.0231 | 0.0551 | 否 | 未达到预设恢复判据 |

恢复判据在分析前声明：先出现扰动，再出现连续 50 步满足配对位置误差增量≤0.05 m、姿态增量≤0.10 rad 的窗口。未满足时恢复时间为 null；没有扰动则另记 no_detected_disruption。μ=0.3 的变化较轻，不能把一个偏严格、有限时长的恢复判据直接等同于控制失败。

不切换 control repeat 在切换后的最大位置误差差异约 1.48e-6 m，远小于 μ=0.15 的变化。低摩擦条件下动作、关节速度、base angular velocity 和 projected gravity 均有明显响应。但动作发生变化只支持反馈响应，不证明 Policy 识别出了 μ。未观察到恢复全局跟踪的证据，也不能据此证明其完全没有局部调整能力。

脚部接触状态和最后物理子步的接触参数已记录；尚未实现可靠的接触点滑动速度，因此不报告滑移速度结论。

## Part C

冻结 Policy，仅拟合独立 Ridge 诊断回归器，alpha=100 预先固定。按父运动分组做 3-fold CV，同一父运动的所有片段与摩擦条件均在同一折。训练集内标准化，比较同一批时间样本；每片段取五个 μ 的共同存活前缀，去掉前 0.5 秒，步长 5。共有 5565 个帧样本、60 个 Episode、10 个独立父轨迹组。5565 个相关帧不代表 5565 个独立样本。

| 输入 | Episode 等权 MAE | RMSE | 相对训练均值基线的 R² |
|---|---:|---:|---:|
| Single frame | 0.2627 | 0.3145 | -0.1036 |
| 10-frame history | 0.2780 | 0.3402 | -0.2920 |
| Training-mean baseline | 0.2560 | 0.2993 | 0 |

图中的误差条按父轨迹等权统计，数值与表中的 Episode 等权统计略有差别。本次探针未显示跨轨迹泛化的摩擦预测能力，也未显示十帧优于单帧。观测差异本身存在，但这不保证一个轻量线性探针能从未见轨迹中可靠辨认摩擦。样本少、高维输入、接触激励与共同前缀筛选都限制结论；不能把负结果解释为历史完全不含信息或模型实现错误。

## 博士面试可用表述

1. **“Friction robustness is metric-dependent: failure flags can hide substantial global tracking drift.”** 配合 01 和 06 图，给出 60 个 Episode 的实际样本数。
2. **“In a strictly paired walking trial, a sudden drop to μ=0.15 increased mean post-change position error by 0.53 m relative to control, without triggering a fall termination.”** 配合 03 图，明确只有一条片段通过严格配对门限。
3. **“Source-grouped diagnostic probes did not establish friction identification or an advantage from ten-frame history.”** 配合 05 图，展示负结果和 mean baseline。
4. 可以说明方法上的发现：必须验证真实接触摩擦和数值容量，否则可能把仿真伪差错当成 Policy 缺陷。

仍属于假设、不可宣称已证明的内容：Policy 显式或隐式识别了摩擦、history 改善控制、缺少 ξ 导致失败、模型存在结构缺陷、对全部 unseen LAFAN1 的普遍 OOD 鲁棒性。需要独立运动扩展、历史消融和更多控制变量才能研究这些机制。checkpoint 本身也不足以独立认证原始预训练数据划分；本实验的 unseen 沿用此前 LAFAN1 评测语境。

## 图表与复现

优先使用 `figures/03_sudden_change_errors.png/pdf`、`01_tracking_score.png/pdf` 和 `05_history_friction_inference.png/pdf`。另有 failure rate、policy / observation changes、fixed root error 图，共六组 PNG/PDF。

代码目录的 README 给出所有采集、容量复核和分析命令。保留的原始 NPZ、独立 CSV/JSON、技术失败记录、源码 / checkpoint / 轨迹哈希及 `validation.json` 支持复查。分析命令：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash my_work/experiments/exp_008_friction_adaptation/run.sh analyze
OMP_NUM_THREADS=2 .venv/bin/python my_work/experiments/exp_008_friction_adaptation/verify.py
```
