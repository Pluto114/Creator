# 原图杆观测与逐段证据：重现说明

这是一套离线研究原型，输出有限中心线候选和拒绝原因。它还不是正式 `PatchResult`，不改原点云，不估计真实表面半径，也没有新增插件操作。

## 数据与边界

运行根目录为 `D:/Creator-newage`。沿用现有环境：

```powershell
Set-Location D:\Creator-newage
. .\scripts\Enter-CreatorEnvironment.ps1
```

前置输入是 [此前对照](thin-pack-controls.md) 中的原始 RGB、4 份完整图估计相机预测、3 份 oracle 预测，以及 `thin-line-controls-v2-20260914` 的不可变简单曲线。模型权重、原始数据和完整输出不进 Git。脚本核对 RGB、模型预测、报告、推理清单及旧曲线身份，禁止覆盖已有 run 目录。

相同的两个人工对应条带继续作为输入；条带范围不是准确端点，也没手工标出缺口。新检测器假设杆大致竖直，扫描双边缘、保留中点与宽度，再用每行最多一票的稳健拟合得到图像线。全局拟合有歧义时不选取其中一条冒充确定结果。

模型原生 K 通过精确半像素 resize 逆映射回原图整数像素中心。oracle 的 K 先在副本上由 edge 转 index。原 NPZ 始终保留。普通候选用估计相机，oracle 是有真实相机的特权诊断，二者不能混成一个平均成绩。

## 先冻结，再评分

先运行两套事先固定的协议。稠密版只把 `row_stride` 从 4 改成 1：这个改变由完美解析直杆的采样失配测试决定，发生在本轮真值评分之前，其他阈值不变。

```powershell
backends/da3/.venv/Scripts/python.exe scripts/run_rod_evidence.py fit
backends/da3/.venv/Scripts/python.exe scripts/run_rod_evidence.py fit --run-id rod-evidence-dense-v1-20260916 --protocol configs/rod_evidence_dense_v1.json
```

每套运行比较 7 份相机预测 × 2 个目标 × 4 个对照，共 56 份候选。三个距离容差只是重复评分，不是新的独立样本。

| 对照 | 作用 |
|---|---|
| `legacy_occupancy` | 直接复用上一轮亮暗脊线与位置投票输出 |
| `paired_occupancy` | 换双边缘观测和稳健2D拟合，仍用旧分段投票 |
| `paired_geometry` | 要求真实像素附近有支持，并检查重投影及留一视图稳定性 |
| `paired_full` | 在上一项基础上允许明确的背景样假设投反对票 |

后两项至少需要 4 个可用图像线视图，每个三维采样点至少有 3 个不同视图支持。拒绝时实际输出为空；`shadow_segments` 仅留作查错，不当成成功候选评分。背景 ORB 匹配目前仅是诊断，既不修改相机，也不参与接受决定；接受检查实际来自目标线重投影和留一视图稳定性。

`absent` 仅表示搜索窗足够平坦。它也可能来自平坦遮挡、未解析细杆或低对比杆，不能仅凭 RGB 静默确认杆不存在。负票对照隐含“未遮挡且目标可被当前曝光/分辨率分辨”的假设；未知、歧义和缺测不投负票。离散取样也不能保证发现任意小的真实缺口。

## 追加评测视角

```powershell
backends/da3/.venv/Scripts/python.exe scripts/export_thin_heldout.py
```

`configs/thin_heldout_v1.json` 指定三个开发条件各自 `−22.5° / +22.5°` 的新视角，RGB、相机、去重几何、中心线可见标签和真实缺口标签全部放入 `data/eval_gt/thin-heldout-v1-20260916`，没有模型输入目录。导出器先回归旧五个角度的相机，再以 Blender 独立射线检查投影与可见标签。

这些视角不参与检测、拟合或阈值选择。它们仍属于同一个资产，不能用来证明跨对象泛化。

两套候选完成后，才进入独立真值阶段：

```powershell
backends/da3/.venv/Scripts/python.exe scripts/run_rod_evidence.py evaluate
backends/da3/.venv/Scripts/python.exe scripts/run_rod_evidence.py evaluate --run-id rod-evidence-dense-v1-20260916
backends/da3/.venv/Scripts/python.exe scripts/plot_rod_evidence.py
backends/da3/.venv/Scripts/python.exe scripts/plot_rod_evidence_overlay.py
backends/da3/.venv/Scripts/python.exe scripts/summarize_rod_evidence.py
```

估计相机只用原始五视图的相机中心拟合一次 Sim3，同一 job 所有方法共用；绝不用杆真值优化对齐。oracle 保持原尺度和坐标。3D 指标同时报告 2/5/10 cm 下的恢复、精度、缺失长度、错误新增长度及带端点保护的缺口误覆盖。

保留视角采用有限线段投影，在可见真值上计算 2/5 px 召回、距离和按投影长度加权的近似精度；不把 GT 不可见样本删除后跨洞连线。近/远裁面异常与出画面拒计数单列。缺口同时报告整段内部和按当前像素容差保护两端后的误覆盖。空候选的恢复为 0、精度无定义；缺口误覆盖为 0 不代表它解决了问题。

## 检查与输出

```powershell
reconstruction/.venv/Scripts/ruff.exe check .
reconstruction/.venv/Scripts/python.exe scripts/check_scaffold.py
reconstruction/.venv/Scripts/python.exe -m pytest reconstruction/tests -q
backends/da3/.venv/Scripts/python.exe -m unittest discover -s tests -q
```

解析测试覆盖中心偏差、真实缺口、纹理歧义、缺测/遮挡、重复视图、退化相机、LOO拒绝、像素映射和稀疏采样失配。轻量 NumPy 测试进入 Windows/Linux CI；I/O runner 的解析适配检查复用本机已有 Pillow/OpenCV 环境。

原始候选在 `.runtime/experiments/<run_id>/`；真值评分在 `data/evaluation/<run_id>/summary.json`。协议、源文件与输入身份随运行冻结，任何新试验使用新 run ID。实际结论见 [9月16日报告](experiments/2026-09-16-rod-evidence.md)，过程见 [当天日志](journal/2026-09-16.md)。
