# 配对细杆的第一轮定量诊断

这个入口把“点被过滤了”和“点的位置不对”分开记账。它属于G0开发实验，还没有实现共同曲线读出、中心线恢复率或错误连接率。

## 运行

在 `D:/Creator-newage` 中，先复用已安装环境：

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
& .\backends\da3\.venv\Scripts\python.exe scripts\run_thin_pack_baseline.py infer --run-id my-baseline-v1
& .\backends\da3\.venv\Scripts\python.exe scripts\run_thin_pack_baseline.py evaluate --run-id my-baseline-v1
```

配置位于 `configs/thin_pack_baseline_v1.json`。默认使用v2配对RGB、LARGE固定权重、估计相机、五个数值排序视角；六条件均运行504，砖块最细另运行756。没有给模型真实相机，也没有调用修复算法。

推理结果和冻结配置位于 `.runtime/experiments/<run-id>/`；评测结果位于 `data/evaluation/<run-id>/`。两阶段可以分开运行。修复评测代码后若要重新评价同一预测，使用新的输出身份：

```powershell
& .\backends\da3\.venv\Scripts\python.exe scripts\run_thin_pack_baseline.py evaluate --run-id my-baseline-v1 --evaluation-id my-baseline-v1-eval2
```

已有输出目录不会覆盖。模型进程的请求只列RGB文件、哈希和推理参数；评测阶段才读取GT。原始NPZ在评测前后核对哈希，不能原地改深度或置信度。每个计划任务都保留状态；失败不能从表里删掉。

## 怎么对齐

先由估计外参和GT外参分别求五个相机中心，再用等权重最小二乘拟合一个正尺度相似变换：统一旋转、平移、缩放。变换只由相机决定，过滤前后共用，不用目标杆做ICP或事后最佳拟合。

相机位于同一平面不是失败条件；近乎共线才会使旋转不稳定。检查中心点第二/第一奇异值比，不足0.001时明确标为alignment_degenerate，三维误差不可测，保留过滤统计。相机拟合RMSE、相机跨度、朝向误差，以及每次拿掉一台相机重新拟合后对它的预测误差，全部单独记录。RMSE超过相机跨度10%会标记poor_camera_fit，不偷偷重选几个好相机来拟合。

这一步消除了整体尺度与坐标原点的任意性，**不会证明模型本来就有准确米制尺度**。

## 每张账到底在算什么

| 字段 | 实际含义 | 不能解释成什么 |
|---|---|---|
| retained_fraction | GT指定像素中，有多少预测点经过该过滤规则仍保留 | 杆恢复率、几何准确率 |
| scaled_camera_z_abs_error_m | 使用相机拟合尺度后的预测Z与同像素GT中心Z之差 | 已消除相机误差的纯深度模型误差 |
| paired_pixel_world_point_displacement_m | 同图像像素对应的预测点与GT交点，经共同相机对齐后的世界位置差 | 最近表面距离、中心线误差、Chamfer或拓扑正确率 |

原始预测、默认40百分位过滤、固定1.05阈值共用同一个对齐变换。默认阈值按照本地上游实现，在五视图全部置信度上取40/90百分位并夹住1.05；比较用大于等于。若很多置信度相同，保留比例未必恰好60%。本诊断不改写天空深度或背景置信度，也不采样点云来计算指标。

几何误差仅在保守单表面区域作为主要诊断：strict_rods和strict_background。mixed_rod_center_diagnostic中的中心射线答案单独列出，明确带有RGB混合像素歧义。像素中心未命中杆但子像素命中杆的区域只统计保留/可见性，不把背景中心深度当成杆深度。空区域填null并给原因；最细杆可能没有strict像素，这不能填成0误差。

过滤后误差变小，也可能只是删掉了难点。因此每个误差分布始终附带原始可用数、保留数和保留比例；逐视图、逐杆数据都保留。

## 那个讨厌的半像素

GT使用目标网格的K_index。模型点云严格遵循固定上游的整数像素网格和原生K，不擅自把模型K减0.5来追求更好分数。真实相机输入涉及的约定转换留给单列的oracle_camera诊断。

上游保存processed_images时还会把归一化张量转回uint8，直接截断会让一些通道比resize结果小1。核对图像时复现上游的float32归一化、float64还原和uint8截断，要求逐值一致；不把正常量化误差当成错位，也不靠放宽像素偏移来通过检查。

## 还没覆盖的事

这七次运行只是在同一个资产上定位问题，不支持泛化或自研方法优越性结论。局部裁剪、简单线拟合、真实相机诊断、共同曲线读出、保留测试对象和实际修复仍需要后续实现。先把这轮原生诊断看明白，再决定优先解决深度、相机、过滤还是多视图对应。
