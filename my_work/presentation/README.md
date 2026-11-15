# SuperMimic / realmimic 实验结果与 PPT 资料入口

这是本项目 Experiment 001–009 的阅读索引。先读本文件，再查原始报告与统计。
社区实现与预训练权重来自 [realmimic](https://github.com/kevinhenryzz/realmimic)，
原方法为 SuperMimic。个人工作是复现、诊断实验和 Exp009 的新增模块；不要将原方法或社区模型归为个人原创。

## 实验结果地图

| 实验 | 实际工作与结果 | 证据入口 |
|---|---|---|
| 001 | 社区权重完整 LAFAN1：40条，score 0.659233，失败14条；历史本地评测导入 | [评测JSON](../outputs/exp_001_lafan1_full/lafan1_full.json)、[README](../experiments/exp_001_lafan1_full/README.md) |
| 002 | 同40条动作切为882个独立10秒片段：score 0.828295，失败37个；不是同一长episode | [评测JSON](../outputs/exp_002_lafan1_seg10/lafan1_seg10.json)、[README](../experiments/exp_002_lafan1_seg10/README.md) |
| 003 | 分析001/002的协议、失败类型与配对运动；不新增仿真 | [报告及图表](../outputs/exp_003_failure_analysis/REPORT.md) |
| 004 | Walking/Dancing/Fall-and-Get-Up 三个案例的仿真录像与可视化；视频留在本地，截图已纳入仓库 | [实验入口](../experiments/exp_004_motion_visualization/README.md)、[截图目录](../outputs/exp_004_motion_visualization/renders/) |
| 005 | 冻结世界模型，3条选定动作，一步与1/4/8/16/32步开环误差诊断 | [报告与数值](../outputs/exp_005_world_model_prediction/REPORT.md) |
| 006 | 社区权重 BONES-SEED 129,491条：score 0.954480，12失败；训练相关数据，不是held-out | [报告](../outputs/exp_006_bones_benchmark/REPORT.md) |
| 007 | 从随机初始化完成86k训练，约10.96小时训练循环；LAFAN1：个人训练0.638998 /16失败，对照0.682738 /13失败 | [训练报告](../outputs/exp_007_full_training/full_seed0/TRAINING_REPORT.md)、[LAFAN1](../outputs/exp_007_full_training/full_seed0/evaluation/LAFAN1_REPORT.md)、[BONES](../outputs/exp_007_full_training/full_seed0/evaluation/BONES_REPORT.md) |
| 008 | 冻结社区权重：固定摩擦60个episode、在线切换及历史探针；实际接触参数验证；未证明历史可识别摩擦或在线适应 | [中文完整报告](../outputs/exp_008_friction_adaptation/REPORT_ZH.md) |
| 009 | 新增Physics Encoder和有界残差模块；社区Policy/WM冻结；最终80,862更新。10摩擦点×两模型×40动作，以及随机摩擦320个episode；没有稳定总体提升 | [本目录统计](results.json)、[固定摩擦原报告1](../outputs/exp_009_history_physics_adaptation/evaluation/sweep_5k_10k_20k_80862/analysis/RESULTS_ZH.md)、[原报告2](../outputs/exp_009_history_physics_adaptation/evaluation/final_vs_community_new_frictions/analysis/RESULTS_ZH.md)、[随机切换](../outputs/exp_009_history_physics_adaptation/evaluation/random_friction_final_vs_community/analysis/RESULTS_ZH.md) |

不同实验中的社区成绩来自不同运行，不能相互替换。Exp007 BONES社区基线明确复用Exp006；
Exp009的固定摩擦基线是本次两个批次实测，不来自网站或Exp008；随机摩擦的不切换对照另行实测。
Exp008使用10秒片段及不同求解器设置，不能与Exp009完整动作成绩直接拼接为同一协议。

## Exp009：重点结果

### 固定摩擦：每个模型、每个条件40条完整动作

表格为 `平均归一化回报 / 失败数`。＊表示超出训练摩擦范围[0.3,1.2]。

| μ | 社区原权重 | 最终80,862权重 |
|---|---|---|
| 0.15＊ | 0.2303 / 37 | 0.2118 / 38 |
| 0.2＊ | 0.3218 / 34 | 0.3527 / 32 |
| 0.3 | 0.5058 / 22 | 0.5106 / 22 |
| 0.4 | 0.6071 / 16 | 0.6350 / 17 |
| 0.5 | 0.6747 / 12 | 0.6657 / 13 |
| 0.7 | 0.6767 / 12 | 0.6735 / 13 |
| 1.0 | 0.6508 / 15 | 0.6843 / 12 |
| 1.2 | 0.6732 / 13 | 0.6766 / 11 |
| 1.5＊ | 0.6678 / 14 | 0.6485 / 14 |
| 2.0＊ | 0.6256 / 16 | 0.5721 / 19 |

数值源：[统一CSV](fixed_friction_comparison.csv)，每行含原始评测批次。
上述800个episode是已有两个批次中这两个模型的子集，不是再次统一重跑。
局部改善不等于整体鲁棒性、范围外泛化或在线适应改善。
配对区间见[CSV](fixed_friction_paired_effects.csv)；区间按40条动作bootstrap，
不代表仿真重复性区间，且未作多条件选择校正。

### 随机切换：每3秒Uniform(0.2,1.5)，首3秒μ=1.0

40条动作×3条配对随机序列×两模型=240个随机episode，另加80个不切换对照，合计320。
三条序列不构成120条独立动作；不允许按时间帧计算独立样本量。

| 条件 | 社区原权重 | 最终80,862权重 |
|---|---|---|
| 不切换对照（各40 episode） | 0.6637 / 13失败 | 0.6768 / 12失败 |
| 随机切换（各120 episode） | 0.6145 / 55失败 | 0.5981 / 56失败 |

最终减社区的配对回报差为−0.0164，按动作bootstrap的95%描述性区间
[−0.0418,+0.0049]，包含0。未显示最终模型的稳定优势，也不能确定其稳定更差。
仅76/240个随机episode通过严格切换前配对门限；两模型共同通过的子集覆盖10条动作。
全样本相对各自对照的损失差偏负，但严格配对子集区间包含0，不作机制性定论。
详见[随机切换报告](../outputs/exp_009_history_physics_adaptation/evaluation/random_friction_final_vs_community/analysis/RESULTS_ZH.md)。

### Physics Encoder输出

输出被结构限制在[0.3,1.2]，范围外不能准确回归真实值。
例如真实μ=0.2时按动作等权平均预测约0.641；真实μ=2.0时约0.877。
预测随环境变化不等于准确参数识别，也不证明其帮助控制。
[预测统计及MAE](encoder_outputs.csv)。

## 推荐用于英文博士面试PPT的图表

| 图表 | 推荐英文主题 | 必须保留的限制 |
|---|---|---|
| [固定摩擦跟踪](01_fixed_tracking.png) / [PDF](01_fixed_tracking.pdf) | Friction robustness depends on the operating regime | 40 motions/condition；两批次单次评测 |
| [固定摩擦失败率](02_fixed_failure.png) / [PDF](02_fixed_failure.pdf) | Failure flags and tracking quality capture different outcomes | 失败不代表所有漂移，得分与失败数可能方向不同 |
| [配对效果与区间](03_fixed_paired_effects.png) / [PDF](03_fixed_paired_effects.pdf) | Local gains do not establish broad generalization | 描述性动作bootstrap；非重复仿真区间 |
| [训练诊断](04_exp009_training.png) / [PDF](04_exp009_training.pdf) | Longer training did not yield consistent real-simulation gains | 想象rollout reward/MAE不是验证集性能 |
| [Exp008在线切换](../outputs/exp_008_friction_adaptation/figures/03_sudden_change_errors.png) | Sudden low friction can produce drift without a fall | 主要机制结论仅一条严格配对动作 |
| [Exp008历史探针](../outputs/exp_008_friction_adaptation/figures/05_history_friction_inference.png) | Grouped probes did not establish friction identification | Ridge负结果不证明历史完全无信息 |
| [随机切换性能](../outputs/exp_009_history_physics_adaptation/evaluation/random_friction_final_vs_community/analysis/01_random_friction_performance.png) | No consistent advantage under repeated friction changes | 40动作、每模型120随机episode；对照另算 |

图表均有PDF版本。英文叙述可以突出实验设计、真实接触验证、协议审计和诚实负结果，
不能写成新模型已经提升在线适应或优于社区。研究计划中的新网络/VLM方向属于未来假设。

## 给GPT的PPT任务文本

可以把以下文字与仓库链接或本地资料包一起提供：

> 请基于 `my_work/presentation/README.md`、`results.json` 及其链接的实验报告，
> 制作英文博士面试PPT。建议12–15页：研究问题、社区方法与个人贡献边界、
> 复现基线、失败分析、世界模型诊断、完整训练复现、固定摩擦、突然切换、
> 历史诊断、Physics Encoder设计、随机摩擦结果、局限与下一步。
> 每个结果页注明实验ID、样本数、指标口径与仓库相对路径。
> 不编造提升，不混用不同批次基线，不把训练reward当验证结果，不把反馈响应当物理识别。
> 明确标注“已验证观察 / 探索性结果 / 未验证假设”。使用现有PNG/PDF及CSV制作图表，
> 给出逐页标题、要点、图表路径与英文讲稿。若无法访问某个文件，应指出缺失，不补造数字。

## 发布与文件保留

GitHub保留：实验代码、README、汇总CSV/JSON、审计记录和PNG/PDF图表。
本地保留但不上传：数据集、所有模型权重、NPZ/NPY原始轨迹、视频、TensorBoard、
逐步训练metrics、逐帧预测JSONL、评测groups明细，以及不完整重复/可行性运行。
不删除上述本地文件。它们的路径可能出现在报告中，但文件本身不会在GitHub提供。

本目录 `source_manifest.json` 记录构建资料时输入文件的SHA256，包括留在本地的训练明细。
重建表格、图表和结构化汇总（CPU，不启动仿真）：

```bash
.venv/bin/python my_work/presentation/build_results.py
```

本目录的 `ppt_results_bundle.zip` 是本地生成的报告/图表资料包，不纳入Git历史。
公开GitHub链接不意味着GPT已自动读取全部文件；请明确提供这个索引或资料包，
并要求其在生成前核对能够访问的证据。
