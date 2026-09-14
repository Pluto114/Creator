# 严格配对细杆数据包 v2

此导出器是 G0 的独立、可重跑实验入口，不声称实现了架构文档中的正式 CaseManifest / Evaluation DTO 或修复算法。唯一实际项目根为 `D:/Creator-newage`。

## 重现

在项目根运行 PowerShell：

```powershell
# 单视图小样例；BundleId 必须使用一个不存在的新名字。
.\scripts\Export-ThinPack.ps1 -BundleId my-paired-pilot -Cases brick-texture-thinner -Angles 0
# 六条件 × 五个数值排序的角度，共30张。
.\scripts\Export-ThinPack.ps1 -BundleId my-paired-full
```

入口复用 D 盘 Blender 5.2.1、DA3 环境中的 Open3D / NumPy / OpenCV / Pillow / Matplotlib，以及本地固定 DA3 预处理源码；没有下载、环境安装或模型推理。启动器设置进程缓存和临时目录到 D 盘，Blender 使用独立配置和 `--background --factory-startup --disable-autoexec`。不得把脚本用于任意透明、动态或修改器场景：它只接受已核实的这个不透明静态资产。

新包已存在时拒绝覆盖。失败保留部分产物与日志，`status.json` 标记 failed；只有 complete 包可用于后续实验。保存配置、生产脚本副本及哈希、上游预处理和几何源码、Git HEAD/dirty 状态、环境版本和全部产物哈希。不保存改动到源 .blend；导出前后都校验源文件 SHA-256。

## 物理隔离与目录

- `data/inputs/<BundleId>/`：模型输入，只含六组 RGB PNG 和 `manifest.json`；按显式 frame_order 读取。清单没有真值相机或几何引用。RGB 是同次渲染 Combined 通道经过源 AgX 显示变换后的 RGB8 PNG。
- `data/eval_gt/<BundleId>/`：评测专用。`export_config.json`、`render_manifest.json`、`provenance.json`、`artifact_hashes.json`、`validation_summary.json` 和各条件的网格、中心线、缺口、逐视图真值。目录隔离是实验边界，并非操作系统权限沙箱；后续模型调用只授予输入目录，不扫描父目录。
- `.runtime/paired-export/<BundleId>/`：请求与 Blender 日志。
- 历史 `.runtime/inputs/test_pack`、`.runtime/experiments`、`data/eval_gt/thin-pack-v1` 全部保留。新 RGB 与历史 RGB 逐图比较仅用于来源核对，不将历史预测重新绑定到新输入。

各条件的 `mesh.npz` 是同一次参数设置下的评估后世界三角网格；包括墙、地面和杆。仅在射线计算中去重完全相同的 Cylinder.003/Cylinder.002，源渲染保留重复物体。`geometry.json` 导出真实中心线端点、尺寸、重复映射和真实缺口。

每视图包含：

- `camera.json`：原生1920×1080的内外参。
- `blender_depth_raw.exr`：同次渲染导出的32位原始深度通道；不作为插值后的精确像素中心真值。
- `blender_ray_probes.json`：Blender 自身 scene.ray_cast 的独立核对样本，覆盖墙、地、杆。
- `native/`：原始像素尺寸的几何真值。
- `da3_504/`、`da3_756/`：实际504×280、756×420的几何真值、准确预处理 RGB 以及 `image_transform.json`。这些 RGB 是评测侧核对副本。
- `validation.json`：坐标、深度、缺口、DA3真实预处理实现对照、特殊像素统计及历史图片比较。
- 部分视图的 `verification.png`：人工视觉核验用图，不参与模型输入。

## 坐标、深度和标签

世界采用 Blender 轴；单位为米（源场景 Metric，scale_length=1）。外参为列向量 `X_cv = world_to_camera_cv @ X_world`，CV 相机 x右、y下、z前。数组索引为 `[v,u]`，图像从左上向下。`K_edge` 的像素中心为 `(u+0.5,v+0.5)`；`K_index` 的像素中心为 `(u,v)`，主点比 K_edge 小0.5。不能重复翻轴或混用两种 K。

主要真值从同一评估后网格的最近可见三角形射线命中导出；阴影、亮线、砖缝不会生成杆标签。射线起点遵守相机 near plane，far plane 后命中无效。32位 mesh/raycast 的数值误差通过 Blender ray_cast 和独立解析平面求交检查，不把数值浮点结果声称为无限精度。

| 数组 | dtype / 形状 | 含义 |
|---|---|---|
| `depth_z.npy` | float32 H×W | 像素中心射线的第一可见表面相机Z，米；无命中为NaN |
| `ray_distance.npy` | float32 H×W | 同一交点到相机中心的欧氏距离，米；无命中为NaN |
| `pixel_sample_depth_min/max.npy` | float32 H×W | 中心与4×4子像素命中的相机Z范围，用于核对渲染通道采样；混合像素可能横跨多个表面，不能当作单表面误差条 |
| `surface_id.npy` | uint32 H×W | 0无命中，1–7杆段，100墙，101地面 |
| `rod_id.npy` | uint32 H×W | 0非杆，1–5五根整杆，6带真实缺口的杆 |
| `rod_visible_center_masks.npy` | bool 6×H×W | 通道0–5分别为杆ID1–6的可见像素中心掩码；已处理遮挡 |
| `segment_coverage_counts.npy` | uint8 7×H×W | 各杆段在规则4×4子像素射线中的可见命中数，除16得到采样覆盖率 |
| `sample_mixed.npy` | bool H×W | 至少一个子像素的表面ID与中心不同 |
| `hit_valid.npy` | bool H×W | 中心射线有相机裁剪范围内的有效命中 |
| `strict_depth_eval_valid.npy` | bool H×W | 保守的单表面深度可评区，规则见下文 |
| `blender_depth_agrees.npy` | bool H×W，仅native | 原始EEVEE通道与中心几何Z在规定容差内一致；不是GT正确性的唯一判据 |
| `blender_depth_within_sample_footprint.npy` | bool H×W，仅native | EEVEE Z位于同像素采样深度范围加1毫米余量内；验收只在strict单表面区域使用 |

1–5分别对应 Cylinder、Cylinder.001、Cylinder.002（含重复.003）、Cylinder.004、Cylinder.005；杆ID6的两段为 Cylinder.006（surface_id6）和 Cylinder.008（surface_id7）。缺口本来存在，约0.619米，记录精确端点和 `intentional_gap_do_not_connect`，不得作为模型断裂或强制补全目标。

## 抗锯齿、亚像素和不可评区

RGB 保持源EEVEE采样/滤波设置，深度和硬ID明确表示像素中心交点。混色像素没有唯一与全部RGB颜色成分一致的表面深度。原生 strict mask 要求中心命中、周围2像素内ID稳定、周围2像素内4×4样本全部与各自中心一致；图像边缘也排除。该规则为本资产的保守评测政策，不是通用渲染器抗锯齿的数学等价实现。

覆盖率是有限几何采样，**不是精确面积，也不是EEVEE的抗锯齿权重**。子采样可能漏掉非常窄的可见片段，不能将0覆盖率证明为不可见。0°视图504尺寸另存8×8检查 `coverage_check_504_8x8.npy`（除64），记录4×4与8×8差异，并断言4×4漏掉而8×8发现的位置不进入strict深度区。需要精确面积指标时必须另行增加解析覆盖率或收敛性标准。

最细档杆常常没有strict深度像素，这是成像可观测性限制，不是删掉困难样本的评测策略。所有中心标签、中心深度、可见掩码与覆盖率仍完整保留；后续评测必须报告排除数量，并用适合混合/亚像素观测的几何或覆盖率指标单列分析，不能用剩余背景像素分数宣称细杆恢复准确。

深度通道用已知5米平面、4米遮挡物与无命中背景实测辨别 camera_z / ray_distance。本机源场景 EEVEE 原始Z与精确中心求交存在约毫米级差异，平面上的有效采样偏移也被观察到；具体渲染内部原因未完全定位。5毫米中心差异阈值只作诊断，远处斜面仍可能超过该值。验收要求strict单表面像素的原始Z落在该像素中心与4×4样本的Z范围内，额外允许1毫米浮点/光栅余量，并另行完整报告中心差异。边缘可显著不一致并明确标记。**这些渲染通道容差不是几何真值的误差声明**：用于评测的Z来自准确中心射线，另以1e-4米容差对照 Blender ray_cast。原始EXR的无命中哨兵实测并记录，规范数组使用NaN和false有效掩码。

## DA3预处理

固定本地DA3源码采用两次INTER_AREA缩放：1920×1080→504×284→504×280，或1920×1080→756×425→756×420。输出RGB与真正 InputProcessor 输出的归一化张量逐值比对。索引坐标采用半像素映射 `u_target = sx*(u_source+0.5)-0.5`，边界坐标直接乘sx。保存逐步尺寸与两种矩阵；目标K由同一变换计算。

目标深度和硬ID均从目标像素中心对应的源连续射线重新求交，**没有对GT深度或ID做双线性插值，也不把最近邻源像素冒充准确目标采样**。低分辨率strict mask还要求两次面积重采样中所有有正权重的原始像素均属同一strict表面，并与目标中心ID一致。保留混合像素的中心答案，但不将其作为唯一RGB深度的可靠标签。

本地上游在传入内参时直接缩放K的行，而反投影使用整数arange像素网格，两者存在半像素约定需要适配：本包同时存储物理正确的K_edge/K_index，并实测上游缩放行为。未来真实相机诊断必须单列 oracle_camera，显式处理这个差异，不能将真值相机混入普通输入或直接沿用未经修正的上游K。

## 核验与可重现边界

验证覆盖固定源哈希、原核实几何/相机复核、Blender投影、独立Blender射线ID/深度、解析平面、反投影/回投、缺口射线、深度类型校准、无命中、真实DA3预处理和分辨率支持域。8项回归测试包括遮挡、no-hit/far clip、亚像素无中心命中、半像素映射、两段缩放尺寸与混合像素排除。

背景/尺寸变化来自同一资产，仍不构成六个独立对象。GPU渲染不能假定跨版本/驱动逐字节一致；新运行建立新输入身份并记录与旧图差异，参数可重现不等于承诺所有环境位级一致。正式评分和任何模型/修复效果属于下一阶段。
