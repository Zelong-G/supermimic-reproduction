# Experiment 004：MuJoCo Motion Visualization

使用社区权重 `weights/realmimic_g1_86k.pt`，观察 Policy 跟踪 LAFAN1 Reference 的成功和失败案例。不训练模型，不修改社区代码。

**三个案例已完成录制和视频导出，12 项 CPU 测试通过。现有视频可直接查看，无需重复录制或导出。**

## 1. 检查输入

在 VS Code 远程终端执行；后续命令均从项目根目录运行：

```bash
cd /home/stud/zheze/storage/user/realmimic
bash my_work/experiments/exp_004_motion_visualization/run.sh check
```

检查权重、动作文件和历史结果是否存在。此步骤不加载模型、不启动仿真。

三个案例已配置在 `cases.json`。下表是历史 Benchmark 结果，不是本次回放结果：

| 动作 | 作用 | 完整序列结果 | 分段失败数 |
|---|---|---|---|
| `walk1_subject1` | 正常成功案例 | 成功 | 0/26 |
| `dance2_subject4` | 完整失败、分段通过 | 第 3046 步，60.92 秒失败 | 0/23 |
| `fallAndGetUp1_subject1` | 早期失败案例 | 第 399 步，7.98 秒失败 | 8/17 |

## 2. 打开第一个回放

预处理结束后执行：

```bash
bash my_work/experiments/exp_004_motion_visualization/run.sh viewer \
  --case fallAndGetUp1_subject1 --run-id view_001
```

1. 等终端显示服务地址。
2. 打开 VS Code 的 **Ports／端口**面板，转发远程端口 `8080`。
3. 在本地浏览器打开转发地址，通常为 `http://127.0.0.1:8080`。
4. 点击 **Play** 开始。实体机器人是 Policy，蓝色半透明机器人是 Reference。
5. 使用 **Pause** 暂停、**Step** 单步、**Reset Environment** 从头回放。
6. 一次回放结束后自动暂停并保存结果；终端按 `Ctrl+C` 退出。

找不到端口面板时，按 `Ctrl+Shift+P`，搜索 `Ports: Focus on Ports View`。端口被占用时，在命令后加 `--port 8081`，并转发 `8081`。

这是浏览器 Viser 界面；原生 MuJoCo 窗口需要服务器图形会话，不能仅靠端口转发显示。

## 3. 观察失败前状态和另外两个案例

退出上一个回放后，按需运行以下命令，**每次只运行一条**。

在早期失败前暂停，再用 Step 单步查看：

```bash
bash my_work/experiments/exp_004_motion_visualization/run.sh viewer \
  --case fallAndGetUp1_subject1 --pause-step 394 --run-id near_failure_001
```

查看舞蹈案例，在历史失败前约 1 秒暂停：

```bash
bash my_work/experiments/exp_004_motion_visualization/run.sh viewer \
  --case dance2_subject4 --pause-step 2996 --run-id dance_view_002
```

查看正常行走：

```bash
bash my_work/experiments/exp_004_motion_visualization/run.sh viewer \
  --case walk1_subject1 --run-id walk_view_001
```

暂停步数只是观察位置，不会跳过仿真；新回放可能提前或延后失败。**失败后环境会自动重置，重置姿态不是失败姿态。**

Viewer 可以一直使用同一个 `--run-id`。每次启动会在该动作目录下自动创建 `session_001/`、`session_002/` 等独立目录；旧结果（包括旧版直接保存在动作目录下的 JSON）保留不动。终端会打印本次输出路径。开头的 `torch.jit.script` 弃用警告不影响这一行为。

## 4. 录制三个案例的状态

先录制早期失败案例，确认成功后，再依次执行另外两条：

```bash
bash my_work/experiments/exp_004_motion_visualization/run.sh record \
  --case fallAndGetUp1_subject1 --run-id capture_001

bash my_work/experiments/exp_004_motion_visualization/run.sh record \
  --case dance2_subject4 --run-id capture_001

bash my_work/experiments/exp_004_motion_visualization/run.sh record \
  --case walk1_subject1 --run-id capture_001
```

每个案例从头运行，到真实失败或完整结束。此步骤保存状态，不生成视频。

## 5. 导出 PPT 视频和截图

三个案例均录制成功后执行：

```bash
bash my_work/experiments/exp_004_motion_visualization/run.sh render \
  --case all --recording-id capture_001 --run-id ppt_001 --seconds 12
```

生成 1280×720、16:9、50 FPS 的 MP4，以及每个案例的三张代表性截图。成功案例取开头 12 秒；失败案例取实际失败前最多 12 秒。录制不足 12 秒时保留实际长度。

若只需要截图，将 `--case all` 改为一个动作名，并使用新的 `--run-id`，在末尾加 `--screenshots-only`。若 EGL 或编码报错，保留错误信息，可先用浏览器回放配合本地屏幕录制。

## 6. 查看输出并核对结果

所有输出位于：

```text
my_work/outputs/exp_004_motion_visualization/
├── preflight.json
├── viewer/<run-id>/<motion>/session_<编号>/episode_000.json   # 实际终止时间及历史对比
├── recordings/capture_001/<motion>/
│   ├── summary.json                          # 录制结果、误差汇总及历史对比
│   └── raw/                                  # 场景、轨迹、逐帧 Root 状态
└── renders/ppt_001/<motion>/
    ├── render_manifest.json
    ├── screenshots/frame_*.png
    └── videos/tracking.mp4
```

核对 `summary.json` 或 `episode_000.json` 的 `comparison`：`exact_outcome_match` 表示成功／失败状态及失败步数一致；`mismatch` 表示有差异。保留差异，不调整参数强行匹配。

将实际命令、失败时间、差异和视频观察补充到本 README。优先选取能清楚展示参考与实际运动偏差的截图放入 PPT。

## 结果解释与重复运行

- 复用社区 Viewer、Benchmark Recording 和 `tools/render_rollouts.py` 的渲染接口；保留真实终止条件。单环境回放可能与原 Benchmark 的未知并行规模产生差异，需同时检查 overflow 记录。
- Root 位姿均在同一去环境原点的世界坐标系中，四元数为 wxyz。位置误差为欧氏距离；姿态误差为四元数最短旋转角。
- Benchmark 失败条件是相对参考的高度误差 **> 0.25 m** 或重力方向 z 分量差 **> 0.8**；后者不是弧度。参考动作本身倒地不等于跟踪失败。
- 录制只保留动作执行前的采样，不含最终终止状态；失败时间采用真实终止步数 × 0.02 秒，不能由最后一张截图推断，也没有失败后画面。
- Viewer 可复用同一 `--run-id`，每次启动自动分配独立 session 子目录。录制和渲染仍禁止覆盖相同模式、动作和编号的输出，重跑需使用新 `--run-id`；渲染时 `--recording-id` 必须对应实际录制编号。
- 原始轨迹和视频被 Git 忽略，小型 JSON 报告和代表性截图可以提交。不执行 push。

## 已有实际结果

来源：`recordings/capture_001/<motion>/summary.json`；视频位于 `renders/ppt_001/<motion>/videos/tracking.mp4`。

| 案例 | 实际结果 | 与历史比较 | Overflow 步数 | 视频时长 |
|---|---|---|---:|---:|
| `walk1_subject1` | 完整成功 | 一致 | 0 | 12.00 秒 |
| `dance2_subject4` | 59.68 秒失败 | 比历史早 1.24 秒 | 10 | 12.00 秒 |
| `fallAndGetUp1_subject1` | 7.98 秒失败 | 一致 | 0 | 7.98 秒 |

每个案例另有三张截图。舞蹈案例存在时间差异和容量溢出，目前不推断因果。

## PPT：10 秒舞蹈跟踪片段

直接使用 `renders/ppt_dance_8to18s_001/dance2_subject4/videos/tracking.mp4`。从已有 `capture_001` 轨迹选取 8–18 秒（跳过开头静止部分），500 帧，1280×720，50 FPS，H.264 MP4，按真实速度播放。实体机器人为 Policy，蓝色半透明机器人为 Reference。

生成命令（再次导出时换新的 `--run-id`）：

```bash
bash my_work/experiments/exp_004_motion_visualization/run.sh render \
  --case dance2_subject4 --recording-id capture_001 \
  --run-id ppt_dance_8to18s_001 --start 8 --seconds 10 --presentation
```

`--presentation` 使用简洁字幕；完整实验对比仍保存在 manifest 和原录制 summary 中。此片段 Root 平均位置误差约 8.89 cm、平均姿态误差约 5.52°，不代表整段舞蹈通过 Benchmark（完整录制在 59.68 秒失败）。建议 PPT 标题写“舞蹈动作跟踪：10 秒片段”，插入视频后设置自动播放，可按需循环。
