# DA3 适配器骨架

当前能执行 `creator-da3 --help`、`--version`、`status`。`run --request ...` 明确返回非零；没有加载模型、下载权重或写出预测。

本包目前不声明上游 DA3 依赖，避免将未经验证的安装组合伪装成可运行环境。G0 阶段固定上游提交与权重，完成 Windows / Python 3.11 / CUDA 推理验证后，再更新依赖和锁文件。

```powershell
uv sync --project backends/da3
uv run --project backends/da3 creator-da3 status
```

`runner.py` 是文件协议入口，`adapter.py` 负责未来模型加载，`preprocess.py` 记录实际图像映射，`serialization.py` 写原始预测包。核心规范化发生在另一环境；这里不生成 Blender 预览或修复几何。
