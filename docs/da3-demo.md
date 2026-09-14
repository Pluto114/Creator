# DA3 最小可视化实验

在 PowerShell 运行（本机默认使用已下载的官方 SOH 两张示例图和 BASE 模型）：

```powershell
& 'D:\Creator-newage\scripts\Demo-DA3.ps1'
```

成功后终端输出 `DEMO_READY=.../demo.html`。双击 HTML，用支持 WebGL 的浏览器打开，不需要服务器或网络。默认从第一张照片的估计拍摄位置和方向观察；拖动旋转、滚轮缩放，按钮可恢复拍摄方向、偏转 20°或显示全场景。相机位置标记可从图例打开。网页的视场角不完全等于照片，因此另附使用相机内参绘制的静态拍摄视角和偏转 15°检查图。

换成自己的照片，或用 LARGE 比较：

```powershell
& 'D:\Creator-newage\scripts\Demo-DA3.ps1' -Images 'D:\photos\1.jpg','D:\photos\2.jpg','D:\photos\3.jpg' -Model large
```

照片准备：同一静止物体，移动相机拍 3–5 个有重叠的视角；第一轮用光线稳定、纹理清楚、不透明的物体。保留原始图片，默认处理长边为 504。本演示限制 2–5 张，未做大批量显存管理。两个相邻视角无法覆盖物体背面。

DA3 预测深度、置信度和相机参数；固定上游代码将像素反投影到共同空间。浏览器和 GLB 各展示最多 10 万个采样点，并过滤低置信区域；`prediction.npz` 保留未经显示过滤的数组。实际阈值写入 `report.json`，不能将阈值或保留点数理解为重建准确率。

查看器将坐标统一变换到第一相机的 glTF 朝向、居中并等比缩放。这个过程不改变点之间的几何关系，不修补缺口，也不裁掉远处背景。静态图使用 3×3 像素点斑和深度遮挡处理，不生成连续表面。原视角看起来像照片，并不能单独证明预测深度正确，还需看偏转视角和独立几何评估。

复用一次成功推理，只重新生成演示页面：

```powershell
. 'D:\Creator-newage\scripts\Enter-CreatorEnvironment.ps1'
& 'D:\Creator-newage\backends\da3\.venv\Scripts\python.exe' 'D:\Creator-newage\scripts\demo_da3.py' --reuse-run 'D:\Creator-newage\.runtime\demo\20260913T134632.908266Z'
```

复用会复制原始 NPZ，记录来源和 SHA-256，不重新推理。Python 入口也支持 `--process-res` 和 `--use-ray-pose`；修改它们需要作为单独实验记录，不能与只修改显示混为一谈。

这是带颜色的点云，不保证完整封闭表面，也不保证单位是米。没有观测到的背面、细杆、反光区域可能缺失或出错。照片/深度页中的深度预览由上游导出，左半是图像、右半是深度配色。

输出在 `.runtime/demo/<UTC时间>/`，每次单独创建并记录成功或失败。默认样例依赖本机 `.local/setup/Depth-Anything-3/assets/examples/SOH`；其他机器可传自己的 `-Images`。模型、缓存、结果均留在 D 盘，尚未接入正式 Creator 任务协议。

源码入口为 `scripts/demo_da3.py`，通过 `smoke_da3.run` 完成固定权重校验和推理，`pointcloud_preview.py` 只处理显示。点云转换调用固定上游的私有工具函数，升级 DA3 时需要重新核对。这是观察基线的实验，不作为 Creator 的研究贡献。此次显示问题的证据和验证范围见 [显示诊断记录](da3-display-diagnosis.md)。
