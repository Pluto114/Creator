# Creator reconstruction core

独立 Python 3.11 核心，当前是 **项目骨架**。CLI 可以显示帮助、版本和实现状态；照片导入、基础重建、局部修复、导出和评测均未实现。任何这些请求都会明确失败，不生成假快照或假成功记录。

## 开发环境

从项目根目录执行：

```powershell
uv sync --project reconstruction --locked --group dev
uv run --project reconstruction --locked creator --help
uv run --project reconstruction --locked creator --version
uv run --project reconstruction --locked creator status
uv run --project reconstruction --locked pytest reconstruction/tests
```

已生成真实 `uv.lock`，并确认本机 Python 3.11 和 uv 可运行。锁文件验证了依赖解析；骨架检查不会运行数值算法或模型，因此不能据此宣称 CUDA、DA3 或完整几何流程已通过验收。

核心依赖按照 [ADR 0002](../docs/decisions/0002-reconstruction-stack.md) 声明 NumPy 2、Pydantic 2、SciPy、OpenCV headless、Pillow 与 trimesh；当前轻量 CLI 只使用标准库，不提前导入它们。DA3/PyTorch 和 Blender 的 bpy 不属于此环境。

## 固定命令入口

```text
creator --help
creator --version
creator status
creator run --request PATH
creator case create --request PATH
```

`status` 输出的是当前安装版本的**能力状态**，不是某个作业的 RunStatus。尚未提供作业查询或后台执行。JSON 请求目前仅做 UTF-8 / JSON 对象入口检查，不宣称完整 v1 契约校验。有效 JSON 也会返回 `E_NOT_IMPLEMENTED`，退出码 3；命令行或入口文件错误返回 2。不会创建输出目录、日志、运行记录或模型文件。

在尚未安装科学计算依赖时，可通过标准库检查当前入口：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "reconstruction/src")
python -m creator_recon --help
python -m unittest discover -s reconstruction/tests -v
```

使用 Python 3.11。上述环境变量只用于当前开发终端；正式安装依靠包入口，不依赖工作目录。

## 边界

- `contracts/` 暂时只包含任务枚举、取消/进度协议和协议设计说明。完整 DTO、schema 校验、内容哈希与黄金样例属于下一阶段，不用不完整字段冒充稳定协议。
- `application/` 固定用例入口并消费共享取消/进度协议，未实现动作立即抛出明确异常。
- `domain/` 保留相机、图像映射、只读快照与补丁组合的模块职责；尚无数值算法。
- `refinement/` 提供可替换入口协议与只读上下文容器；多视图候选算法仍未实现。
- `infrastructure/` 与 `adapters/` 只声明后续职责，不启动模型、下载依赖或写入半成品。

下一个可验收交付是独立相机投影检查与照片→规范基础快照→Blender 预览的最小闭环，见 [系统设计](../docs/architecture/01-system-design.md)。本包可导入不代表该闭环已经完成。
