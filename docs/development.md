# 骨架开发说明

当前从 `0.1.0.dev0` 骨架开始。架构文档描述目标系统；源码中的职责模块和明确拒绝执行的入口不代表目标能力已经完成。

## 环境

三个独立 uv 项目：`reconstruction/`、`backends/da3/`、`experiments/`。Python 目标为 3.11，不把 Blender 自带 Python 用作核心解释器。实验包通过本地路径依赖核心，依赖方向不可反转。

使用各项目提交的锁文件同步。核心锁定数值依赖，DA3 的 `inference` extra 锁定独立的官方模型运行栈。先按 [环境说明](environment.md) 配置项目内缓存及环境，再运行检查。依赖成功解析、CLI 可以启动与模型推理成功是不同的验证结果。

## 无模型检查

```powershell
uv run --project reconstruction --locked ruff check .
uv run --project reconstruction --locked python -m pytest reconstruction/tests
python scripts/check_scaffold.py
python scripts/build_addon.py
```

`check_scaffold.py` 只使用 Python 3.11 标准库：检查源码语法和 TOML，再在隔离的临时工作目录里调用各 CLI。它验证未实现的入口不会伪造结果，不验证相机、CUDA 或算法质量。

插件真实检查需要 Blender：

```powershell
blender --background --factory-startup --python-exit-code 1 --python blender_addon/tests/smoke_blender.py
```

测试期间建议将 `BLENDER_USER_CONFIG`、`BLENDER_USER_SCRIPTS`、`BLENDER_USER_EXTENSIONS` 指向临时目录，避免访问个人插件和扩展缓存。首次目录需自行建立；不要在个人正在编辑的 Blender 进程中执行测试。

CI 在 Windows 和 Linux 上检查轻量入口、核心测试和 Python 包构建；不下载模型权重，不将 CI 绿灯描述成模型运行或研究方法通过。

## 开发顺序

1. 实现最小输入和相机契约，导出真实 JSON Schema 与微型样例。
2. 验证坐标与图像变换，再接 DA3 原始预测。
3. 实现快照、受控测试补丁和 Blender 预览。
4. 验证进程取消与恢复，完成 G0 闭环。
5. 用真实候选方法和公平对照判断是否存在可重复增益。

每完成一项，就更新根 README 的实际能力表和相应关键测试。不要删除未完成提示后直接返回成功，也不要把 mock 或人工构造的测试补丁当算法输出。

## 仓库内容

`data/`、`runs/`、`cache/`、`models/`、`.local/`、`.runtime/` 和虚拟环境只在本地使用。`scripts/build_addon.py` 的 ZIP 输出到被忽略的 `dist/`。插件打包只包含插件自身的 `.py` 文件，不包含模型、核心环境、测试或个人配置。

本轮替换旧主分支文件采用普通后继提交，历史 C++ 原型仍可通过前一提交检出。不同依赖项目分别维护锁文件，不建立囊括所有模型的根 uv workspace。
