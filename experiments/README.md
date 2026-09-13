# 独立实验包

当前为骨架：`creator-eval --help`、`--version`、`status` 可用；`run --request ...` 明确失败，不生成假分数或空成功报告。

```powershell
uv sync --project experiments
uv run --project experiments creator-eval status
```

本包通过本地路径依赖核心，核心不能反向导入本包。GT 与保留评测视图只进入这里，不进入修复请求。

未来模块对应 [评测设计](../docs/architecture/05-evaluation-and-delivery.md)：数据分组与真值、固定全局对齐、共同几何读出、指标和报告。`protocols/` 保存冻结协议；`synthetic/` 保存合成样本生成脚本。当前尚无真实评测协议、样本集或算法质量结论。
