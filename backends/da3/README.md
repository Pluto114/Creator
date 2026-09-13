# DA3 隔离后端

模型运行环境已经安装并验证；Creator 正式文件协议入口仍为骨架。当前 `creator-da3 --help`、`--version`、`status` 可执行，`run --request ...` 明确返回非零，不会伪造预测。

`inference` extra 固定官方 DA3 源码、PyTorch CUDA wheel 及独立的 NumPy 1.26 环境；不要与核心 NumPy 2 环境混装。完整安装与本机 GPU 证据见 [环境说明](../../docs/environment.md) 和 [验证记录](../../docs/environment-verification.md)。

从仓库根目录运行：

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
uv sync --project backends/da3 --locked --extra inference
uv run --project backends/da3 --locked --extra inference creator-da3 status
```

保留 `--extra inference`，否则 uv 同步会移除模型依赖。独立上游推理通过 `scripts/smoke_da3.py` 验证，尚未接到 `creator-da3 run`，因此当前能力状态仍显示 `inference_ready: false`。

`runner.py` 是未来文件协议入口，`adapter.py` 负责模型加载，`preprocess.py` 记录实际图像映射，`serialization.py` 写原始预测包。核心规范化发生在另一环境；这里不生成 Blender 预览或修复几何。