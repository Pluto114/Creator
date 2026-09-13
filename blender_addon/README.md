# Blender 插件骨架

当前版本只有可注册的插件、3D Viewport 侧栏和本机路径配置。**尚未实现照片导入、DA3 推理、细杆恢复、ROI、预览导入或任务启动。** 界面没有假重建按钮，也不会联网、启动外部 Python、安装包或改写模型文件。

目标宿主是 Blender 5.2 LTS。本目录采用 legacy add-on ZIP 布局；不是 Blender Extensions 分发包，暂不提供 `blender_manifest.toml`。许可证继承项目根目录的最终决定，不在此额外声明另一份许可证。

## 安装与配置

1. 使用项目打包工具生成插件 ZIP，ZIP 根目录须直接包含 `creator_recon/`，其下有 `__init__.py`；不要把整个项目 ZIP 当插件安装。
2. 在 Blender 的 Preferences → Add-ons 中选择 Install from Disk，选择 ZIP 并启用 **Creator: Thin Structure Reconstruction**。
3. 打开 3D Viewport 右侧栏（N 键）的 **Creator** 页，或在插件偏好设置中配置两个路径。
4. `Core Python` 指向未来独立核心环境的 Python 可执行文件；`Project root` 指向含有 `reconstruction/pyproject.toml` 的项目根目录。两者默认为空，要求本机绝对路径。
5. **Check configured paths** 只检查文件是否存在及项目布局；检查通过不表示 Python 可执行、依赖已安装或算法可用。更改路径后需重新检查。

插件使用 Blender 自带 Python，只引用 `bpy` 和标准库。不将外部 `creator_recon` 核心包、torch、Pydantic 或 DA3 安装到 Blender 中。插件目录与核心包同名但属于不同解释器；禁止在同一个 Python 进程中把二者混合导入。

## 已实现与待实现边界

| 位置 | 当前实现 | 后续对应设计 |
| --- | --- | --- |
| `__init__.py` | 注册/卸载 Blender 类与 Scene 属性，注册失败回滚 | 完整宿主生命周期入口 |
| `properties/` | `CreatorPreferences`、只用于显示的 `CreatorSceneProperties` | 可序列化案例/版本引用，不保存进程句柄 |
| `operators/` | `CREATOR_OT_check_configuration`，只读检查 | 显式导入、计算、取消、接受/撤回操作 |
| `ui/` | `CREATOR_PT_reconstruction` | 照片 ROI 控件、状态与版本比较 |
| `bridge/` | `validate_configuration()` | 受归属约束的 `RunProcessBridge` 与轻量文件协议 |
| `session/` | 仅边界说明 | `AddonSession`、`VersionController` |
| `importing/` | 仅边界说明 | `PreviewImportSession`，分块主线程导入 |

后续契约见 [任务与 Blender 设计](../docs/architecture/04-runtime-and-blender.md)。未实现模块不提供返回假成功的空类或进程启动占位代码。当前 Scene 属性仅是配置检查反馈，不是核心任务状态，且设置了 `SKIP_SAVE`。

## 独立注册验证

在项目根目录的 PowerShell 中运行，替换 Blender 路径为自己的安装位置：

```powershell
$creatorBlender = '<absolute path to blender.exe>'
& $creatorBlender --background --factory-startup --python-exit-code 1 --python .\blender_addon\tests\smoke_blender.py
```

验证脚本检查两次完整的注册/卸载，要求不残留 RNA 类或 Scene 属性、不新增场景对象、不导入 torch/Pydantic。后台模式不能验证面板视觉布局、鼠标交互或实际 ZIP 安装，应另外在 Blender 界面中检查；它也不能证明尚未实现的推理/取消/预览流程正确。
