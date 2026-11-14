# Experiment 006：BONES-SEED Benchmark Reproduction

使用社区权重 `weights/realmimic_g1_86k.pt`，调用原始 `scripts/eval_benchmark.py`。只评测，不训练、不修改社区模型。

**BONES-SEED 属于训练相关数据评测，不是未见动作泛化实验。**

**rand2000、hard 和 full 均已完成；8 项静态测试通过。汇总见下方实际结果及输出 REPORT.md。**

## 1. 数据检查

从项目根目录执行：

```bash
bash my_work/experiments/exp_006_bones_benchmark/run.sh check
```

检查预处理进程、过滤清单、三个动作列表、所有保留动作的文件存在性，以及 16 个分布采样 NPZ 的维度和 50 Hz 帧率；不是逐文件完整解码验证。记录 GPU 可用显存。任一条件不满足，评测不会启动。

本次检查时已有 142,220 个 NPZ、过滤清单和训练库；仅缺 `data/bench/`。已用社区 `tools/make_benchmark_sets.py` 生成最后三个小型列表，没有重新转换、下载或复制数据。

| 文件 | 数量 | 社区 MD5（本地一致） |
|---|---:|---|
| `manifest_filtered.csv` | 129,491 条 keep=1 | `5bc2aae2efc6f986ab4cf7adbb4d8ffc` |
| `rand2000.txt` | 2,000 | `28dfbc14f02146cc05fe62c794d89f01` |
| `hard.txt` | 130 | `b931bb29b8c786bfdfa34b57c0832caf` |
| `full.txt` | 129,491 | `e588d04ab59d948b86347088dff357f0` |

## 2. Random 2000

```bash
bash my_work/experiments/exp_006_bones_benchmark/run.sh rand2000
```

完成后检查 `rand2000_run.json` 的 `status`、结果数量和 overflow，再进入下一步。运行日志在 `logs/rand2000.log`。

## 3. Challenging Motion

```bash
bash my_work/experiments/exp_006_bones_benchmark/run.sh hard
```

130 条高难动作覆盖：side rolls、flips、fire-in-the-hole turns、crossed legs、broad jumps。它不是 Table A.8 全部类别的集合。

## 4. Full Benchmark（确认后再执行）

前两个阶段完成、检查且经你确认后，再手动运行：

```bash
bash my_work/experiments/exp_006_bones_benchmark/run.sh full
```

没有自动串行启动 full 的入口。社区完整集参考：129,491 条、平均 tracking score 0.9545、失败 10 条。子集结果不能直接当作完整集复现结果。

## 5. 分析与输出

每个阶段成功后自动更新分析，也可以单独运行：

```bash
bash my_work/experiments/exp_006_bones_benchmark/run.sh analyze
```

```text
my_work/outputs/exp_006_bones_benchmark/
├── preflight.json
├── rand2000.json / hard.json / full.json    # 原始逐动作结果；仅实际运行后生成
├── rand2000_run.json / hard_run.json / full_run.json
├── summary.csv
├── category_summary.csv
├── failures.csv                            # 本地失败动作名称、时间、分数及 overflow
├── REPORT.md
├── figures/                                # 英文 PNG + PDF
└── logs/                                   # 本地日志
```

主分数使用 `mean_norm_return`，包含提前终止惩罚；frame-weighted score 单独记录。失败率为失败动作数 / 评测动作数。按社区 `motion_group()` 统计类别失败分布。缺失的阶段不填零、不编造结果。

## 参数与保存规则

- 默认同一 checkpoint、raw policy 权重、`--num-envs 1024 --chunk-size 2000 --device cuda:0`。
- 实际环境数由社区脚本取动作数与请求数的较小值：rand2000 为 1024，hard 为 130。Full 按 chunk 大小及最后一块实际数量计算。
- 如需调整资源：在命令后加 `--num-envs 512 --chunk-size 1000`。实际命令、请求/有效并行数、MD5、checkpoint SHA-256、软件版本、GPU 和耗时均保存在 `*_run.json`。
- 不覆盖已有结果；已完成时直接运行 `analyze`。全局运行锁阻止本实验的两个阶段同时启动。不同阶段会验证 checkpoint 哈希一致。
- 大型逐动作 JSON 和日志保留本地；小型 `failures.csv` 按用户要求提交，供逐动作失败分析。汇总表、运行参数和图表可提交。原数据和权重保持本地，不执行 push。

## 静态测试

```bash
.venv/bin/python -m unittest discover \
  -s my_work/experiments/exp_006_bones_benchmark -p 'test_*.py'
```

测试覆盖结果汇总、动作覆盖、失败时间、无效列表、数据未完成时阻止启动和防覆盖；不运行仿真。

## 本次实际结果

| 阶段 | Tracking score | Failed / Total | Failure rate | 记录 overflow 的动作数 | 耗时 |
|---|---:|---:|---:|---:|---:|
| rand2000 | 0.956083 | 0/2000 | 0% | 13（15 个溢出步） | 228.09 秒 |
| hard | 0.744589 | 1/130 | 0.7692% | 3（3 个溢出步） | 36.86 秒 |
| full | 0.954480 | 12/129491 | 0.0093% | 1186（1225 个溢出步） | 2827.59 秒 |

hard 类别失败数：side rolls 0/22、flips 0/54、fire-in-the-hole turns 0/20、crossed legs 1/10、broad jumps 0/24。唯一失败在 10.90 秒触发终止，该动作无溢出；名称见本地 `failures.csv`。

完整集实测得分 0.954480、12 次失败；社区参考为 0.9545、10 次失败，失败数多 2 条。溢出记录保留在原始 JSON 中，主分数未擅自剔除动作。详见 `REPORT.md`、类别表和失败清单。
