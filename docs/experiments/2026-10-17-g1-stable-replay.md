# 当前稳定版本的分阶段独立重复回放 — 2026-10-17

结论：**48/48 行完成，24/24 对重复结果完全相同；48 行均与当前稳定参考的对应配置完全相同。** 四份新 `boundary-patch-1` 均已保存、重开和撤回，基础点云及 point IDs 字节不变。本报告确认这一版本的重复性与保存链路，不自行判定 G1。

## 固定范围与执行

- 新运行：`g1-stable-replay-20261017-001`。
- 上游：`g1-end-to-end-20261013-001` 中 chair01/aframe01 × 两次独立进程的**各自实际** prediction、base、rods、control fits、相机和 RGB evidence；没有借用另一次重复的 native 输出。
- 稳定方法：`creator_eval.rod_boundary_patch`，`boundary-patch-1`；稳定共同读出器：`creator_eval.rgb_bracketed_occupancy_readout:BracketedOccupancyReadout`。与 `boundary-patch-sprint-20261014-001` 的 policy 与共用直接源码哈希一致。
- 每个新进程执行六臂 × 两尺度：`base`、`depth_tls_single_span`、`depth_ransac_single_span`、`baseline`、`cylinder_support`、`baseline_boundary`；尺度为相机跨度的 0.0025 和 0.005。全部 48 行 normal 完成后，单独 evaluate 才读取原始两物体完整 GT 并使用旧评分函数。
- 新 DA3 调用 **0**，新 RGB 推理/相机标定 **0**，新 baseline/control 重建 **0**。复用的是 10 月 13 日四次独立 DA3 的完整上游结果，新执行的是四个独立进程的最后一段。这是 **split-stage replay，不是一次新的全链路端到端重复或新采集数据**。

| 物体 / repeat | 上游进程 PID | 新回放进程 PID | 新进程耗时 s | 行数 |
|---|---:|---:|---:|---:|
| chair01 / 0 | 30632 | 36124 | 23.788 | 12 |
| aframe01 / 0 | 38676 | 9368 | 90.355 | 12 |
| chair01 / 1 | 37268 | 3892 | 23.785 | 12 |
| aframe01 / 1 | 33776 | 37668 | 91.002 | 12 |

四进程并行 normal 总耗时 **97.234 s**；evaluate 函数内耗时 **0.456 s**，不含解释器启动。四个 worker 和 normal/evaluate 命令均正常退出。

## 结果与来源凭据

- 24/24 配对：common segments/state、native segments/state、common/native 全部评分及错误字段逐项完全相同。
- 48/48 行：以上字段与当前 stable60 中对应的两物体 × 六臂 × 两尺度参考完全相同。其余三个夹具不在本次重复范围，没有伪称重跑 60 行。
- 48/48 common 状态 complete；0 个评分错误。此次没有精度变化，也没有调整阈值、核心算法或挑选有利重复。
- 四份新 patch 的 enabled/withdrawn 文件均真正重开；检查完整 base points/IDs 字节和 native segments。新复制 base 的每个文件 SHA 与该 worker 自己的原始 base 相同。
- 新 plan 绑定四份上游 normal/integration、prediction、原始 native fit、patch/base 文件、共用标注及执行源码；新 normal 再绑定四份实际 worker PID/normal/integration。
- 补充来源收据：执行完成后，仅对实际 evidence 使用的 **10 张 RGB** 各计算一次 SHA，全部与 10 月 13 日 plan/direct_hashes 及各自两次 worker 的 frame.rgb_sha256 一致，共 20 个 frame 引用。没有修改冻结执行源码或重跑。完整收据见 `.runtime/work-logs/20261017-g1-stable-replay.json`。

结果 JSON 中 `g1_passed=false` 是 runner 的固定“**本脚本不作 gate 判定**”标记，**不等于最终验收否决**；G1 的汇总判定由主线验收文档负责。本实验仍是已见开发数据，不是独立盲测或新真实采集。

## 命令与产物

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
.\.venv-tools\Scripts\python.exe -B -m ruff check scripts/run_g1_stable_replay.py --no-cache
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_g1_stable_replay.py normal --run-id g1-stable-replay-20261017-001
.\backends\da3\.venv\Scripts\python.exe -B scripts/run_g1_stable_replay.py evaluate --run-id g1-stable-replay-20261017-001
```

该 run ID 已使用，禁止覆盖或重跑。Ruff 在运行前发现一次 import 排序问题，修正后通过；执行之后源码未改，无全量测试。

- 完整 48 行及 24 对逐字段比较：`docs/experiments/results/2026-10-17-g1-stable-replay.json`。
- 执行目录：`.runtime/experiments/g1-stable-replay-20261017-001`，含四 worker stdout/stderr、normal、integration、refinement、保存 patch、plan/manifest。
- 助手：本地 Codex 子代理 `/root/raster_band_tests`；没有调用 DeepSeek。动作与哈希收据保留于 `.runtime/work-logs/20261017-g1-stable-replay.{md,json}`。
