# 独立实验包

正式CLI仍为骨架：`creator-eval --help`、`--version`、`status` 可用；`run --request ...` 明确失败，不生成假分数或空成功报告。独立实验入口已实现于 `scripts/run_thin_pack_baseline.py`，调用本包的 `native_diagnostics.py` 完成相机对齐、过滤保留量和同像素点位诊断，见 [运行说明](../docs/thin-pack-baseline.md)。

```powershell
uv sync --project experiments
uv run --project experiments creator-eval status
```

本包通过本地路径依赖核心，核心不能反向导入本包。GT 与保留评测视图只进入这里，不进入修复请求。

目标模块对应 [评测设计](../docs/architecture/05-evaluation-and-delivery.md)：数据分组与真值、固定全局对齐、共同几何读出、指标和报告。当前原生诊断协议在 `configs/thin_pack_baseline_v1.json`，配对数据保存在独立eval_gt目录。共同曲线读出与正式恢复率仍未实现，不把原生点位诊断当作完整研究评分。

## 后续实测

已完成裁剪、真实相机和两个目标的TLS/RANSAC/RGB交会对照，见[运行说明](../docs/thin-pack-controls.md)。line_controls是无GT拟合及有限段指标辅助模块；当前人工对应的目标曲线诊断不替代正式CurveReadout或完整候选评测。
