# 裁剪、真实相机与简单直线对照

当前结果见[实验报告](experiments/2026-09-14-crop-oracle-line-controls.md)。所有命令在D:/Creator-newage执行，先复用已安装环境：

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
```

## 已有结果与重现身份

本机本轮结果已经存在，下面的导出/推理/拟合命令在原身份上重复执行会拒绝覆盖。完整再现需要新运行ID；输入bundle也需在复制的协议中换新ID。不要删除旧目录来腾名字。

偏心裁剪：
```powershell
& .\backends\da3\.venv\Scripts\python.exe scripts/prepare_thin_controls.py inputs
& .\backends\da3\.venv\Scripts\python.exe scripts/prepare_thin_controls.py truth
& .\backends\da3\.venv\Scripts\python.exe scripts/run_thin_pack_baseline.py infer --protocol configs/thin_pack_controls_v1.json --run-id thin-controls-v1-20260914
& .\backends\da3\.venv\Scripts\python.exe scripts/run_thin_pack_baseline.py evaluate --run-id thin-controls-v1-20260914
```

居中追加对照：prepare的两阶段都增加
`--config configs/thin_pack_centered_controls_v1.json`；
推理使用这个协议及运行ID `thin-centered-controls-v1-20260914`。

两个prepare阶段独立：inputs只读RGB；truth才读取GT。GT生成前核对完整组/帧顺序、冻结裁框、源图片和裁图逐像素对应；生成时按变换后的相机重新打射线，不对深度或ID做双线性插值。

真实相机诊断：
```powershell
& .\backends\da3\.venv\Scripts\python.exe scripts/run_thin_oracle_controls.py --run-id thin-oracle-controls-v1-20260914
& .\backends\da3\.venv\Scripts\python.exe scripts/evaluate_thin_oracle_controls.py --run-id thin-oracle-controls-v1-20260914
```

原图相机作为额外输入只进入此独立运行。worker保存原生输出与相机覆盖前输出，核对尺度方向和像素约定。只对居中全图开放本轮oracle路径，因为上游相机编码忽略主点。自检入口为 `scripts/thin_pack_oracle_worker.py --self-test`，不用载入权重。

直线对照：
```powershell
& .\backends\da3\.venv\Scripts\python.exe scripts/run_thin_line_controls.py fit --protocol configs/thin_line_controls_v2.json --run-id thin-line-controls-v2-20260914
& .\backends\da3\.venv\Scripts\python.exe scripts/run_thin_line_controls.py evaluate --protocol configs/thin_line_controls_v2.json --run-id thin-line-controls-v2-20260914
& .\backends\da3\.venv\Scripts\python.exe scripts/plot_thin_control_diagnostics.py --run-id thin-line-controls-v2-20260914
```

fit读取允许的RGB、对应条带和已有预测，不读杆GT；evaluate才做相机对齐和目标评分。v2协议引用的完整图/裁图/oracle运行必须已完成。复制协议更换这些引用即可评价新运行；已有输出拒绝覆盖。

## 方法含义

- depth_tls_*：相同RGB条带选出模型像素，去掉同视图重复像素，按指定置信过滤后做等权TLS。
- depth_ransac_unfiltered：同一批未过滤点，固定256次/seed0，两点假设，距离阈值是估计相机跨度0.005倍，最小点对跨度0.03倍，至少12点/3视图。共识集再TLS；没有GT阈值或GT挑点。
- rgb_multiview_planes：相同原图脊线拟合2D直线，回投影平面求共同3D直线。普通轨用估计K/pose；oracle轨只用已知K/pose，不用预测深度。
- single_span：按观测投影范围min/max输出一段，可能把真缺口连起来。
- multiview_supported：相机跨度0.002倍的bin，单视图占据膨胀±1bin，至少3视图、至少连续3bin。仅是沿线位置投票，不包含严格的射线距离或对应检验；因此故意保留其失败作为基线。

对每个指定目标，用固定相机变换后的有限段，计算2/5/10 cm三档弧长加权覆盖/精确比例、双向误差、错误预测长度和缺口附近覆盖。空预测R=0、P=null，输出不删；同一集合的共线重叠段拒绝重复计长。它不是通用CurveReadout，也不评分完整点云候选。

原生预测所有点位诊断的strict集合随采样网格改变；不要把不同网格的中位数差当严格逐点配对差。curve轨虽共用两个完整GT目标，但tight/centered裁剪会限制可见范围，截掉的端部仍计入完整目标的缺失。

## 审计产物

每次运行保存冻结协议、脚本哈希/副本、输入/预测身份与失败信息。开发过程中只做记录加固或CLI扩展，没有改写已完成结果。v1/v2共用528条指标完全相同。

`scripts/summarize_thin_controls.py`核对当前16份原生预测/报告及30张原RGB，并生成随仓库分享的数值摘要。它汇总已存在的固定本轮运行；不触发模型，也不把780条指标当独立样本。

本轮图由实际数据生成并查看：RGB选点图展示RANSAC前的原始接受点；数值图中的零缺口覆盖必须结合杆覆盖看。只有两目标、同一资产及五输入视图，不宣称自动跨对象恢复。
