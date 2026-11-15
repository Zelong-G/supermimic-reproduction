# Experiment 007：Full Training Reproduction

目标：从头训练新的 World Model、Policy 和 Critic，复现社区 `sm-train release`。不读取 `weights/realmimic_g1_86k.pt` 初始化模型，不复用 Smoke Test 权重。

**新路径动作库已重建并通过检查，Smoke Test 已通过，Full Training 已启动，尚未完成。日志、显存监控及自动报告已接入；实时进度以本地运行记录为准，不将未完成训练写成已完成结果。**

## 1. 训练实际包含什么

1. 创建模型和优化器；`resume=None`、`init_from=None`。网络按社区规则初始化，包含随机层和 Dynamics 的零初始化输出层。
2. 在 MuJoCo 中采集 **40 个向量化环境步**，填充 Replay Buffer。
3. 前 **500 次迭代**：继续采集数据，只更新 World Model。
4. 后续迭代：先采集数据，再更新 World Model，然后通过暂时冻结参数的 World Model 优化 Policy，最后用 GAE 目标拟合 Critic。
5. 达到 **86,000 次迭代**结束；共 85,500 次 Policy 更新。UTD=0.4，通常每迭代采集 2 或 3 个环境步。

不需要额外预训练 World Model。它在本次训练中先 warm up，再与 Policy 交替更新；不是先完整训练一个 World Model、再单独训练 Policy。

| Release 参数 | 数值 |
|---|---:|
| 环境数 | 4096 |
| World Model batch / 窗口 / 每次更新 mini-batch 数 | 2048 / 8 / 10 |
| Policy batch / rollout horizon | 2048 / 32 |
| Critic epochs / minibatches | 5 / 4 |
| Replay capacity（每环境） | 1000 |
| WM、Policy、Critic 初始学习率 | 0.0002 |
| 学习率衰减迭代 / 最低比例 | 69000 / 0.03 |
| Seed / AMP | 0 / none（不使用 BF16，允许社区 TF32 设置） |

`profiles.json` 以原始 release preset 为基准；full 不覆盖训练配方。Trainer 会自动派生目标维度、compact library 等配置，实际配置保存在运行目录的 `config.json`。

## 2. 检查并修复 Motion Library

从项目根目录运行只读检查：

```bash
bash my_work/experiments/exp_007_full_training/run.sh check
```

原 `data/library/bones_paper.pt` 为约 6.3 GiB，检查时缺少 ZIP/ZIP64 中央目录。新库已在下述独立路径重建并验证；原文件未修改。**此前 Benchmark 读取 NPZ，所以与它成功完成并不矛盾。** 原文件保持不变。

你准备好后，用现有 NPZ 重建到新路径（不是复制 checkpoint，不重新下载）：

```bash
bash my_work/experiments/exp_007_full_training/run.sh rebuild-library
```

该命令调用社区构建脚本，固定写入 `my_work/outputs/exp_007_full_training/prepared_data/bones_paper.pt`，拒绝覆盖已有文件或指向原文件的符号链接。原损坏文件不会被覆盖。已有新库时直接检查，不要重复重建。

这是一次较重的 CPU/磁盘操作，建议放在 tmux 中运行。当前已完成重建，不必重复执行。进度写入 `my_work/outputs/exp_007_full_training/library_build.log`，可另开终端用 `tail -f` 查看。重建后检查：

```bash
bash my_work/experiments/exp_007_full_training/run.sh check \
  --library my_work/outputs/exp_007_full_training/prepared_data/bones_paper.pt
```

必须看到 `full_data_ready: true`。检查覆盖 129,491 个动作的名称、长度、连续帧范围、fp16 71 维形状和 64 个边界采样；不是全量数值/CRC 扫描。该 `.pt` 是参考动作库，**不是预训练模型**。

## 3. 先运行 Training Smoke Test

Smoke 使用过滤后的前 4 个 BONES-SEED NPZ，不依赖损坏的 packed library：

```bash
bash my_work/experiments/exp_007_full_training/run.sh smoke --seed 0
```

仅 12 次迭代：前 4 次只更新 WM，后 8 次更新 Policy/Critic。环境 64、WM batch 32、Policy batch 16、horizon 8；保留 release 网络结构，关闭 torch.compile。其余差异见 `profiles.json`。

通过条件：三组网络参数有限且发生变化、存在有限梯度、优化器步数递增、WM/Policy/Critic 指标有限、checkpoint 可重新加载且参数一致。检查结果：

```text
my_work/outputs/exp_007_full_training/smoke_seed0/smoke_checks.json
```

必须为 `passed: true`。Smoke 不是性能评测，也不证明 full batch 显存一定足够。

## 4. 用 tmux 启动完整训练

完成库修复和 Smoke Test 后，再手动执行；脚本不会自动开始 full：

```bash
tmux new -s sm_full
```

在新会话中进入项目根目录，执行准确命令：

```bash
cd /home/stud/zheze/storage/user/realmimic
bash my_work/experiments/exp_007_full_training/run.sh full --seed 0 \
  --library my_work/outputs/exp_007_full_training/prepared_data/bones_paper.pt
```

它通过 `monitor.py` 调用同一个社区 `sm-train release` 入口，训练参数与原生命令相同；只增强日志，不包装网络、Loss 或优化器。原生等价命令和实际调用命令均保存在运行信息中。没有 `--resume` 或 `--init-from`，也不加载 Smoke 权重。

离开会话：`Ctrl+B`，松开，再按 `D`；可以断开 SSH。重新连接后：

```bash
tmux attach -t sm_full
```

在训练窗口按 `Ctrl+C` 会中断训练，原程序不保证此时额外保存；tmux 也不能防止服务器重启。崩溃恢复需显式使用社区 `--resume`，会加载优化器和采样器，但不恢复 Replay Buffer/RNG，因此不保证逐位连续复现。

## 5. 监控 Loss 和进度

另开终端：

```bash
tail -f my_work/outputs/exp_007_full_training/full_seed0/console.log
```

日志每 50 次迭代报告 `it`、耗时、it/s、WM 误差、想象回报及 overflow。前 500 次没有 Policy/Critic 更新是正常的。首次 torch.compile 可能较慢。

启动 TensorBoard：

```bash
.venv/bin/tensorboard \
  --logdir my_work/outputs/exp_007_full_training/full_seed0/tb \
  --host 127.0.0.1 --port 6006
```

在 VS Code 转发端口 `6006`，打开本地地址。重点查看：

- `wm/loss`、`wm/state_loss`、`wm/online_nse_1step`、`wm/online_nse_8step`。
- `policy/imag_objective`（越大越好；原程序没有单独记录 `policy/loss`）、`policy/critic_loss`。
- `train/episode_norm_return`、`time/env_steps` 和 `sim/overflow_*`。

### 自动报告与 PPT 图表

训练正常结束、抛异常或收到可处理的中断后，会自动分析已有日志。也可以在训练中或结束后手动刷新：

```bash
bash my_work/experiments/exp_007_full_training/run.sh metrics --run-name full_seed0
```

独立分析入口（默认平滑窗口 5）：

```bash
.venv/bin/python my_work/experiments/exp_007_full_training/analyze_logs.py \
  --run-name full_seed0 --smooth-window 5
```

输出五组英文 PNG/PDF，横轴均为 iterations，并标出 full 的 **500 次迭代阶段边界**（Smoke 按自身配置为 4）：

| 文件名 | 原始指标 | 含义 |
|---|---|---|
| `01_world_model_loss` | `wm/loss` | Training Loss |
| `02_world_model_prediction_metric` | `wm/online_nse_8step` | 训练 replay 上的 Prediction Metric，不是独立 Benchmark |
| `03_policy_imagination_objective` | `policy/imag_objective` | Objective，越大越好 |
| `04_critic_loss` | `policy/critic_loss` | Training Loss |
| `05_training_throughput` | 控制台已有 it/s | 日志窗口内的吞吐；缺失时由 TensorBoard 耗时差推导 |

图中保留半透明原始曲线，并叠加最近最多 5 个日志点的尾随算术均值；不插值，在事件文件切换、step 重启或 NaN/Inf 处断开。`--smooth-window 1` 关闭平滑。缺少的指标不画假曲线，前 500 次没有 Policy/Critic 图属于正常情况。

### 日志可靠性与显存

- 复用原 TensorBoard 标量，不在训练中另写一份相同数据。事件每 **15 秒**刷新；原每 50 次迭代的指标计算频率保持不变。
- 新增 `gpu/torch_peak_allocated_mib`、`gpu/torch_peak_reserved_mib`：PyTorch 的进程内峰值，不包含 Warp 的独立分配。
- 每 **30 秒**采样本训练进程的 `nvidia-smi` 显存，包含 Warp 等分配；这是采样峰值，可能漏掉瞬时尖峰，不冒充精确峰值。
- NaN/Inf 标量写入本地异常日志；复用已有梯度范数、overflow 和 unphysical 指标。结束后检查 checkpoint 参数有限性和优化器 step。不会为监控添加评测或改变训练目标。
- 元数据记录起止时间、含启动开销的进程耗时、配置、seed、版本/commit/hash、正常完成/中断状态及初始化/恢复标记。未到目标或中断时，区分最终 checkpoint 的计数和最后已观测计数，不把它们当作未知的最终步数。
- 对 SIGINT/SIGTERM 尽量刷新日志；SIGKILL、断电或尚未到日志点的步骤仍可能丢失部分记录。保存机制不变，不会额外强制生成 checkpoint。
- 当前 full 入口仍只从零启动，不自动 resume。分析会识别配置/checkpoint 的 resume 字段及重复 step，并保留原始历史；若绕过本监控入口直接续训，新增显存指标无法事后补算。
- 不保存环境变量或 Hugging Face Token。原始事件、异常日志和较大的 `scalars.csv` 留在本地；汇总、throughput CSV 和图表可提交。

## 6. 输出和资源

```text
my_work/outputs/exp_007_full_training/
├── preflight.json / full_plan.json / smoke_plan.json
├── prepared_data/bones_paper.pt             # 手动重建后才存在，Git 忽略
├── smoke_seed0/                             # 执行后才存在
└── full_seed0/                              # 执行后才存在
    ├── run_metadata.json / requested_config.json / config.json
    ├── monitor_status.json
    ├── training_summary.json / TRAINING_REPORT.md
    ├── console.log
    ├── ckpt_latest.pt / ckpt_0004000.pt / …
    ├── tb/
    ├── metrics/scalars.csv / throughput.csv
    ├── figures/01_*.png / .pdf … 05_*.png / .pdf
    ├── logs/anomalies.jsonl                  # 只记录实际检测到的异常
    └── evaluation/                          # 训练后评测
```

使用社区 `log_root/run_name` 原生保存路径，不搬运或复制大型 checkpoint。每 4000 次保存 latest 和编号快照；最终 86k 模型是 **`ckpt_latest.pt`**，最后一个定期编号快照是 84k，不会自动生成 `ckpt_0086000.pt`。模型、优化器、采样器和配置均在 checkpoint 中。

当前机器：RTX 4090 24 GiB，PyTorch 2.14.1+cu130、CUDA runtime 13.0、驱动 580.178.04；GPU 空闲。约 235 GiB 可用主存，文件系统约 9.7 TiB 可用（共享空间，不代表独占配额）。每次启动重新记录 seed、软件/GPU、commit、dirty 状态和源码哈希。当前基础 commit 为 `f851af0`，实验文件尚有未提交改动，不能只凭该 commit 重建全部实验脚本。

社区报告同级 GPU 约 10.5 小时；这不是本机实测承诺。预计 22 个 full checkpoint 约占 5–8 GiB，另有日志和编译缓存；建议预留 30 GiB，库重建另需约 6.3 GiB。缓存和首次编译会影响用时/显存；脚本不自动减小 release batch。

## 7. 评测自己训练的模型

与实验 006 已有社区权重结果比较（不重跑社区权重）：

```bash
.venv/bin/python my_work/experiments/exp_007_full_training/benchmark_bones.py rand2000
.venv/bin/python my_work/experiments/exp_007_full_training/benchmark_bones.py hard
```

使用 `full_seed0/ckpt_latest.pt`、1024 个请求并行环境和 chunk size 2000，与实验 006 一致。
脚本校验列表数量、MD5、数据文件及结果动作对应关系；拒绝覆盖已有输出。
JSON、日志、运行元数据及 `BONES_REPORT.md` 保存到 `full_seed0/evaluation/`。
仅重新汇总已有结果时执行：

```bash
.venv/bin/python my_work/experiments/exp_007_full_training/benchmark_bones.py analyze
```

这两个子集属于训练数据相关评测，不代表未见动作泛化；脚本不启动 full。

先确认训练完成，再执行：

```bash
.venv/bin/python scripts/eval_benchmark.py \
  --ckpt my_work/outputs/exp_007_full_training/full_seed0/ckpt_latest.pt \
  --npz-dir data/npz/bones_seed --clip-list data/bench/rand2000.txt \
  --num-envs 1024 --chunk-size 2000 \
  --out my_work/outputs/exp_007_full_training/full_seed0/evaluation/rand2000.json
```

先检查 rand2000，再将列表和输出文件名同时改为 `hard`、`full`。这些是你新模型的评测，不能覆盖实验 006。完整集对比：社区预训练 0.9545/10 failures、本机预训练复现 0.954480/12 failures，以及本次新训练结果。BONES-SEED 不是未见动作泛化评测。

## 静态测试

```bash
.venv/bin/python -m unittest discover \
  -s my_work/experiments/exp_007_full_training -p 'test_*.py'
```

20 项测试通过。测试不会创建 Trainer、启动训练或分配 GPU 模型；日志/绘图测试使用临时目录中的合成事件，测试完成后清理，不当作实验结果。已有输出目录拒绝覆盖；weights、packed library、TB 事件和逐动作评测结果不提交。配置、README、运行信息、小型标量表和分析图表可以提交；不执行 push。
