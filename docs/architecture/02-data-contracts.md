# 02 · 数据契约、坐标与文件协议

日期：2026-09-13。状态：待实现的首版规范；文中的字段和接口不是已存在的代码。

本协议服务于一个明确流程：照片产生不可变的基础重建，修复器产生绑定该基础的局部补丁，核心导出可展示的完整版本，独立评测器判断几何是否改善。任何进程都不能靠猜测文件名、数组顺序或当前 Blender 场景来补齐数据含义。

## 1. 模式来源与各环境责任

唯一模式定义源计划放在 `reconstruction/src/creator_recon/contracts/`，由核心的 Pydantic 2 数据模型定义，导出 JSON Schema 到 `schemas/v1/`。本文是这些模型的设计说明；实施时同一变更必须同时更新模式和对应说明。

| 计划文件 | 公开 DTO / 类型 | 责任 |
| --- | --- | --- |
| `common.py` | `ArtifactRef`、`ArraySpec`、`Provenance`、`ErrorRecord` | 文件身份、数值描述、来源、错误 |
| `case.py` | `CaseImportRequest`、`CaseImportStatus`、`CaseManifest`、`FrameRecord`、`RegionSelection` | 用户输入准备与明确的区域选择 |
| `camera.py` | `CameraRecord`、`CoordinateConvention`、`ImageTransformChain` | 相机、图像映射和尺度 |
| `backend.py` | `BackendRequest`、`BackendPredictionManifest` | 模型环境与核心的文件边界 |
| `snapshot.py` | `BaseSnapshot`、`SnapshotFrame`、`BasePointId` | 归一化后的研究输入 |
| `patch.py` | `PatchResult`、`CenterlineSet`、`PointSet`、`EvidenceRecord` | 局部改动与可追踪证据 |
| `run.py` | `RunRequest`、`RunStatus`、`RunRecord` | 运行输入、进度与最终记录 |
| `preview.py` | `PreviewManifest`、`PreviewVersion` | 核心验证后的 Blender 展示入口 |
| `evaluation.py` | `EvaluationRequest`、`EvaluationReport` | 独立评测入口；指标细节由评测设计规定 |

Blender 与模型适配器不导入核心包，也不安装 Pydantic。它们只读写 JSON、NPY、PNG 等文件，使用少量显式的字段、版本和路径检查。这个轻量 reader 不能宣称完成了全模式或几何验证：模型输出由核心完整校验；Blender 只导入核心发布的 `PreviewManifest`，不直接解释原生模型数组。

核心分三步校验：`parse` 校验结构和枚举，`validate_artifacts` 校验文件与数组，`validate_semantics` 校验坐标、身份和跨字段关系。通过结构校验不代表通过几何准确性评测。

## 2. 公共类型与兼容规则

### 2.1 基本类型

本文 `T?` 表示必须出现、允许为 JSON `null` 的字段；`list[T]` 可以为空，只有标明 `>=1` 时要求非空。若标记“可选字段”，才允许整个键缺省。数值矩阵在 JSON 中按行存储，数学计算采用列向量。

| 类型 | 线格式与限制 |
| --- | --- |
| `SchemaVersion` | 字符串，首版 `1.0.0` |
| `Id` | 大小写敏感 ASCII，正则 `[A-Za-z0-9][A-Za-z0-9_-]{0,95}`；不含目录分隔符 |
| `Hash` | `sha256:` 后接 64 位小写十六进制；原始文件 SHA-256 与内容对象 SHA-256 使用同一种写法，但字段语义不同 |
| `UtcTime` | RFC 3339 UTC 字符串，以 `Z` 结尾 |
| `Matrix3` / `Matrix4` | JSON `number[3][3]` / `number[4][4]`，全为有限值 |
| `Vec2` / `Vec3` | JSON `number[2]` / `number[3]`，全为有限值 |
| `JsonConfig` | 一个具体模型或方法自己的版本化、封闭字段对象；不是任意脚本、可导入类名或表达式 |

每个独立 JSON 文档具有 `schema_version` 和 `document_type`；嵌套对象不重复这两个字段。本文各表除特别注明，均省略这两个公共字段。核心模型采用严格字段与类型验证；意外字段、把字符串当数字、未知枚举均报错。

`1.0.x` 只修正文档或校验错误，不改变既有字段含义；增加字段或枚举需新 minor，改变坐标、单位、身份或必填字段需新 major。首版 reader 只接受自己明确列出的版本，不假设“同 major 自动兼容”。升级通过显式迁移产生新文件与新内容身份，保留旧产物；不就地改旧快照。模式导出文件必须与核心契约测试一起检查，防止文档、模型和 Schema 各自演化。

### 2.2 `ArtifactRef` 与数组

`ArtifactRef` 是文件引用，禁止把绝对磁盘路径直接写进可移植研究产物。

| 字段 | 类型 | 含义与不变量 |
| --- | --- | --- |
| `store` | `case \| snapshot \| run \| evaluation` | 已配置的逻辑存储区 |
| `bundle_id` | `Id` 或不带 `sha256:` 的 64 位摘要 | 存储区下的一次案例、快照、运行或评测目录 |
| `path` | `string` | 包内相对路径，统一 `/`；必须指向普通文件 |
| `sha256` | `Hash` | 被引用文件完整字节的哈希 |
| `byte_size` | `int >= 0` | 完整文件长度 |
| `media_type` | `application/json \| application/x-npy \| application/x-npz \| application/octet-stream \| image/png \| image/jpeg \| text/plain \| text/markdown` | 首版明确允许的格式；binary 仅允许第 9 节固定预览布局；新增格式要更新 reader |
| `array` | `ArraySpec?` | NPY 必需；PNG 与普通 JSON 为 `null`；NPZ 由下述成员表描述 |
| `members` | `map[string, ArraySpec]?` | 仅 NPZ 使用；成员名固定，不含路径；NPY 为 `null` |

`ArraySpec` 字段为 `dtype: string`、`shape: list[int]`、`order: "C"`、`meaning: string`、`unit: string`。`meaning` 是具体契约规定的标识，如 `camera_z_depth`，不能代替字段本身的类型约束。契约规定 dtype、维度和单位，reader 同时核验文件头；不能只相信 JSON 声明。

大数组优先独立 NPY，允许分块和只读 mmap；NPZ 只用于小型成组数组，不能要求将整份大型压缩归档解压进内存。所有 NPY/NPZ 禁止 object dtype 和 pickle。首版持久化类型限定 little-endian `float32`、`float64`、`uint32`、`int64`、`uint8` 和 `bool`；实际 NPY 头允许 bool/uint8 的无字节序形式。哈希按最终文件字节计算，不承诺不同 NumPy 写法具有相同哈希。

存储根由本机可信配置映射，例如 `case → data/cases`、`snapshot → cache/snapshots`、`run → runs`、`evaluation → data/evaluation`。外置数据盘只改变本机映射，不改变研究身份。跨环境启动器向子进程提供本次所需的有限 root bindings；bindings 是运行配置，不参与内容哈希，也不是可执行指令。

`ArtifactResolver` 必须拒绝绝对路径、盘符、UNC、空段、`.`、`..`、反斜杠、冒号和 NUL；解析后的真实路径必须留在指定 bundle 内。Windows 上检查 junction、符号链接和其他 reparse 跳转，不能仅用字符串前缀检查。输入只能位于允许的读区，输出只能位于本任务临时目录。文件哈希和长度均须匹配；大文件可以分块计算。常规重建与修复请求禁止包含 `evaluation` 引用，评测引用仅在 `evaluate` 分派生效。

### 2.3 来源、错误与 JSON 数值

`Provenance` 字段：`producer: string`、`producer_version: string`、`source_commit: string?`、`model_id: string?`、`weight_revision: string?`、`weight_sha256: Hash?`、`source_kind: measured|model_estimated|user_provided|derived|synthetic|gauge_choice`、`parent_artifacts: list[ArtifactRef]`。未获得提交号或权重哈希时明确为空且记录限制，正式冻结实验前补齐。

`ErrorRecord` 字段：`code: Id`、`message: string`、`stage: string`、`retryable: bool`、`details: map[string, JSON scalar]`。错误不传 Python 异常实例或任意对象；详细堆栈写日志。示例错误码：`SCHEMA_UNSUPPORTED`、`ARTIFACT_HASH_MISMATCH`、`COORDINATE_UNSUPPORTED`、`INSUFFICIENT_VIEWS`、`GPU_OOM`、`CANCELLED_BY_USER`。

JSON 禁止 NaN/Infinity，缺失用 `null`。大型数值数组允许规范指定位置的 NaN；禁止 Infinity。原生模型输出可暂存 NaN/Infinity 以保留原始事实，但必须显式标为 native，并由归一化步骤产生有效掩码；不可直接提升为规范化快照。

## 3. 输入：`CaseManifest`、`FrameRecord` 与区域

### 3.0 导入前的 `CaseImportRequest` / `CaseImportStatus`

用户最初只有外部照片，还没有 CaseManifest，因此输入准备不能要求先引用一个不存在的案例包。`creator case create --request <path>` 解析 CaseImportRequest，再调用 `CaseBuilder.create(source_paths, options, output_dir, cancel)`；这是受管理的短期导入进程，不增加第五种 RunKind，也不计作模型推理成本。

`CaseImportRequest` 字段为 `request_id: Id`、`target_case_id: Id`、`capture_group_id: Id`、`description: string`、`static_scene_asserted: bool`、`source_images: list[SourceImage], >=1`、`known_camera_source: CalibrationSource?`、`image_policy: ImageImportPolicy`。这里是唯一允许用户原始绝对照片路径进入的边界文档：`SourceImage` 字段为 `frame_id: Id`、`source_path: string`；必须是用户选定、存在、可读取的普通图片文件，列表顺序就是输入顺序。启动器只授予这些已解析文件的读取范围，不能把其文本解释为命令或任意目录展开。

`CalibrationSource` 字段为 `source_path: string`、`format: creator_camera_records_v1`、`provenance: Provenance`；导入前校验具体 camera、frame 对应、单位和来源。没有有效标定文件时用 null，不能靠照片文件名或相机摆放印象造出 known cameras。来自模型的既有估计仍标 model_estimated，合成相机仍标 synthetic，不能冒称实测。

`ImageImportPolicy` 字段为 `color_conversion: rgb8_srgb_v1`、`exif_policy: apply_declared_orientation`、`preview_max_edge: int>0`、`preview_resampling: string`。颜色或 EXIF 政策改变构成新案例计算输入；仅预览尺寸改变只重建显示派生文件。

`CaseImportStatus` 字段为 `request_id`、`state: queued|running|cancel_requested|cancelled|failed|completed`、`sequence: int>=0`、`progress: number[0,1]?`、`message: string`、`updated_at: UtcTime`、`case_manifest: ArtifactRef?`、`error: ErrorRecord?`。只有完整案例包原子发布后才能 completed；半成品不可选作运行输入。CaseBuilder 负责复制原文件、计算哈希、执行 EXIF/颜色转换、生成预算内照片预览及坐标映射。ProcessSupervisor 负责进程归属与取消；导入进度和耗时保留在导入记录中。

### 3.1 `CaseManifest`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `case_id` | `Id` | 用户可辨认的案例身份；修订不可覆盖原内容 |
| `case_content_hash` | `Hash` | 对本节定义的稳定内容投影计算 |
| `revision` | `int >= 1` | 人类审阅用修订号；不替代内容哈希 |
| `description` | `string` | 静态对象和拍摄条件说明 |
| `static_scene_asserted` | `bool` | 用户声明同一静态对象；不是算法证明 |
| `frame_order` | `list[Id], >=1` | 明确的输入视图顺序，不依赖文件系统排序 |
| `frames` | `list[FrameRecord]` | ID 唯一，集合与 `frame_order` 严格相同 |
| `known_cameras` | `list[CameraRecord]` | 可为空；来源和坐标必须齐全，不补造未知相机 |
| `known_coordinates` | `CoordinateConvention?` | 有已知相机/世界单位时必需 |
| `capture_group_id` | `Id` | 同一物体/拍摄组的稳定关联；切分不能只按照片 |
| `created_at` | `UtcTime` | 不参与内容哈希 |

### 3.2 `FrameRecord`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `frame_id` | `Id` | 稳定视图 ID；改照片内容会改变案例内容哈希 |
| `original_image` | `ArtifactRef` | 用户提供的原始文件，原样保留 |
| `original_size_wh` | `[int>0, int>0]` | 原文件解码后的宽、高，尚未执行 EXIF 方向变换 |
| `exif_orientation` | `int 1..8` | 无有效 EXIF 时明确使用 1，并记入 provenance |
| `oriented_image` | `ArtifactRef` | 执行 EXIF 方向规范化后的 PNG；首版 RGB、uint8、sRGB |
| `oriented_size_wh` | `[int>0, int>0]` | 用户选 ROI 时看到的尺寸 |
| `photo_preview` | `ArtifactRef` | Blender 照片画布用的预算内 PNG，显示派生文件 |
| `photo_preview_size_wh` | `[int>0, int>0]` | 实际预览图尺寸 |
| `preview_to_oriented` | `Matrix3` | 预览像素中心 → oriented 原图像素中心，按真实 resize 规则计算 |
| `exif_pixel_transform` | `Matrix3` | 原像素中心 → 方向规范化像素中心，含旋转/反射/平移 |
| `orientation_provenance` | `Provenance` | 方向来自文件还是明确的人工修订 |
| `mask_policy` | `none \| user_foreground` | 不自动假设已做抠图 |
| `foreground_mask` | `ArtifactRef?` | 如提供，PNG uint8 `[H_o,W_o]`，0/255；坐标为 oriented 图 |

首版不默默将透明背景、16 位输入或 HDR 解释成普通 RGB；输入准备阶段执行一个命名、记录的转换政策，原文件仍保留。颜色处理不是几何真值，转换政策参与输入身份。

### 3.3 `RegionSelection`

顶层字段为 `selection_id: Id`、`selection_content_hash: Hash`、`case_content_hash: Hash`、`regions: list[RegionRecord]`、`operation_log: list[SelectionOperation]`、`created_at: UtcTime`。时间、UI 会话信息不参与选择内容哈希；实际区域、来源和传播方法参与。

`RegionRecord` 字段如下：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `region_id` / `frame_id` | `Id` / `Id` | 区域与视图身份 |
| `coordinate_space` | 固定 `oriented_original_pixels` | 不能存成 Blender 面板缩放后的屏幕坐标 |
| `kind` | `box \| polygon \| mask` | 判别联合，只允许对应一种载荷 |
| `box_xyxy` | `[number,number,number,number]?` | 最小/最大 u、v；仅 box 有值 |
| `polygon_uv` | `list[Vec2]?` | 仅 polygon 有值，至少 3 点，简单多边形 |
| `mask` | `ArtifactRef?` | 仅 mask 有值；PNG uint8 `[H_o,W_o]`，0/255 |
| `origin` | `manual \| propagated \| imported` | 人工选择不等于细结构真值 |
| `parent_region_ids` | `list[Id]` | 传播来源；人工首选通常为空 |
| `propagation_method` | `Provenance?` | propagated 必需；不能伪装成人工直接选择 |

像素中心 `(0,0)`，整个图像的连续边界是 `[-0.5, W-0.5] × [-0.5, H-0.5]`。box 是此连续域中的半开范围 `[umin,umax) × [vmin,vmax)`，宽高必须为正；多边形必须落在域内。采样和 rasterize 的规则固定为“像素中心是否在区域内”，边界按固定半开规则处理。`SelectionOperation` 只存 `operation_id`、`kind`（create/replace/delete）、`region_ids`、`timestamp`、`actor`（user/propagator/importer）、`note`；真正算法输入由最终 `regions` 决定，不能靠重放 UI 日志恢复。

## 4. 相机、预处理与尺度

### 4.1 `CoordinateConvention`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `coordinate_frame_id` | `Id` | 本包共享的世界框架名称 |
| `handedness` | 固定 `right` | 核心世界及相机的手性 |
| `camera_axes` | 固定 `x_right_y_down_z_forward` | CV 相机局部轴 |
| `extrinsic_direction` | 固定 `world_to_camera` | `T_cw` 的方向 |
| `vector_convention` | 固定 `column` | 列向量 |
| `pixel_origin` | 固定 `top_left_pixel_center_zero` | `(u,v)`；数组读取 `[v,u]` |
| `depth_kind` | 固定 `camera_z` | 不是到相机中心的射线距离 |
| `metric_status` | `unknown \| predicted_metric \| calibrated_metric` | 全快照共享尺度状态 |
| `length_unit` | `reconstruction_unit \| meter` | unknown 对应前者，其他两种对应 meter |
| `reference_frame_id` | `Id?` | 采用第一输入视图作为世界原点时记录其 ID |
| `source_to_world` | `Matrix4` | 从明确的模型源世界框架到当前世界框架的全局相似变换 |
| `calibration` | `ScaleCalibration?` | 实测标定才允许 calibrated_metric |

`ScaleCalibration` 字段为 `factor: number>0`、`source_unit: string`、`target_unit: "meter"`、`measurement_description: string`、`measurement_artifact: ArtifactRef`、`provenance: Provenance`。转换后的点、深度和相机平移已经使用同一尺度；消费者不得再乘一次 factor。`source_to_world` 只允许正尺度的刚体相似变换，不允许反射或任意非均匀拉伸。

无外部世界框架时，以 `frame_order[0]` 的相机框架为整个快照的世界框架，而非按模型内部参考视图顺序；通过一个共同变换转换所有视图。模型估计了米制尺度只能标为 predicted_metric，不能写成“已校准”。世界轴不自动代表重力方向。

### 4.2 `CameraRecord`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `frame_id` | `Id` | 对应一张明确视图 |
| `coordinate_frame_id` | `Id` | 引用共享坐标；不允许各视图各自使用不同尺度 |
| `K_oriented` | `Matrix3?` | oriented 原图对应的内参；无法确定时为空 |
| `K_processed` | `Matrix3?` | 规范化深度图/处理图对应的针孔内参 |
| `T_cw` | `Matrix4?` | 世界 → 相机；快照中参与多视图修复的相机不可为空 |
| `distortion` | `DistortionRecord` | 已校正、明确模型或未知近似 |
| `preprocess` | `ImageTransformChain?` | oriented 图与处理图之间的精确映射 |
| `intrinsic_provenance` | `Provenance` | 实测、模型估计、人工或推导 |
| `pose_provenance` | `Provenance` | 不将 gauge choice 写成模型估计 |

`T_cw` 最后一行为 `[0,0,0,1]`；R 正交、det 接近 +1，数值容差在验证配置中固定并记录。内参焦距为正，K 最后一行为 `[0,0,1]`。源数据若带 skew，必须明确支持投影，不能读入后丢弃。深度位于相机前方，即有效像素 z>0。

`DistortionRecord` 字段为 `status: known_none|corrected|known_model|unknown_pinhole_approximation`、`model: opencv_brown|opencv_fisheye|null`、`coefficients: list[number]?`、`calibration_source: ArtifactRef?`、`mapping: ArtifactRef?`。系数顺序遵循具体命名模型并在 schema 注明。corrected 时深度与 `K_processed` 必须对应校正后的图；unknown 不能等同于“证明无畸变”。第一轮合成闭环采用 known_none。

### 4.3 `ImageTransformChain`

字段：`source_frame_id: Id`、`source_size_wh: [int,int]`、`target_size_wh: [int,int]`、`processed_image: ArtifactRef`、`steps: list[ImageTransformStep]`、`mapping_direction: "source_to_target_with_inverse_sampling"`。源是 oriented 图，目标是实际与深度对齐的图。

`ImageTransformStep` 使用 `kind` 判别：

| kind | 必需载荷 | 说明 |
| --- | --- | --- |
| `resize` | `input_size_wh`、`output_size_wh`、`pixel_transform: Matrix3`、`interpolation`、`sampling_rule` | 如 `half_pixel` / `align_corners` / 明确的 library 规则；记录实际矩阵，不能只给目标短边 |
| `crop` | 两端尺寸、`origin_uv`、`pixel_transform` | crop 的平移要同时作用到 K |
| `pad` | 两端尺寸、`padding_ltrb`、`fill_rgb`、`pixel_transform` | 记录填充对几何有效区域的影响 |
| `undistort` | 两端尺寸、`source_model`、`source_K`、`target_K`、`inverse_map: ArtifactRef`、`valid_mask: ArtifactRef` | 非线性映射另存，不能伪装成 3×3 矩阵 |

非线性 inverse map 为 float32 `[H_target,W_target,2]`，每个目标像素对应源 `(u,v)`，无映射位置填 NaN 且 valid mask 为 false；只使用有效坐标。对包含非线性步骤的 ROI 映射，需要实际映射采样或几何投影，不直接乘一个总矩阵。

后端如果另有网络特征分辨率，必须写出深度输出到处理图的实际映射；不能假定所有输出尺寸等于输入。一个来源不明的归一化 K，不能根据“看起来合理”猜单位后进入核心。

## 5. 模型环境边界

### 5.1 `BackendRequest`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `request_id` / `run_id` | `Id` / `Id` | 调用归属 |
| `case_manifest` | `ArtifactRef` | 核心已校验的案例清单 |
| `frame_order` | `list[Id], >=1` | 本次实际输入子集与顺序 |
| `input_images` | `list[ArtifactRef]` | 与 frame_order 一一对应的 oriented 图 |
| `known_cameras` | `list[CameraRecord]` | 没有则为空；oracle 与预测来源不得混淆 |
| `known_coordinates` | `CoordinateConvention?` | 有世界相机输入时必需 |
| `backend` | `BackendSpec` | 明确的模型、代码和参数 |
| `seed` | `int` | 固定随机条件；不承诺 CUDA 完全逐位可复现 |
| `requested_outputs` | `list[depth|confidence|cameras|processed_images]` | 首版不请求无关的高斯/材质 |
| `output_bundle_id` | `Id` | 只允许写本 run 的指定 backend 临时子目录 |
| `input_content_hash` | `Hash` | 不含本次 run ID 的计算输入身份 |

`BackendSpec` 字段：`backend_id: da3|moge3`、`adapter_version: string`、`source_commit: string`、`model_id: string`、`weight_revision: string`、`weight_sha256: Hash?`、`environment_lock_sha256: Hash`、`inference_config: JsonConfig`。DA3 配置必须明确处理尺度/策略、输出过滤是否关闭、精度和相机相关选项；不能把上游 defaults 当成永远不变。MoGe3 仅作为已启用的单图对照配置，不能凭共同 DTO 宣称具备多图外参。

### 5.2 `BackendPredictionManifest`

这是“上游到底输出了什么”的收据，并非 `BaseSnapshot`。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `backend_prediction_id` | `Hash` | 原生输出包的内容身份 |
| `request_content_hash` | `Hash` | 对应 BackendRequest 的稳定计算输入 |
| `backend` | `BackendSpec` | 实际运行配置，不是用户期望值 |
| `frame_order` | `list[Id]` | 必须精确对应请求，或给出显式重排映射 |
| `capabilities` | `BackendCapabilities` | 有哪些原生产物 |
| `source_conventions` | `NativeConventions` | 原始相机、像素、深度和尺度解释 |
| `frames` | `list[NativeFramePrediction]` | 每帧原生数组与真实处理链 |
| `provenance` | `Provenance` | 模型及适配来源 |
| `warnings` | `list[string]` | 限制及异常，不作为未知语义的替代解释 |

`BackendCapabilities` 字段均为 bool：`depth`、`confidence`、`intrinsics`、`shared_multi_view_extrinsics`、`predicted_metric_scale`。能力来自已验证适配器，不由用户随意声明；`shared_multi_view_extrinsics=false` 的多张单图输出不能直接进入多视图修复。

`NativeConventions` 必须具体给出：`camera_axes`、`handedness`、`extrinsic_direction`（world_to_camera/camera_to_world/absent）、`extrinsic_shape`（3x4/4x4/absent）、`vector_convention`、`pixel_origin`、`intrinsic_encoding`（pixel/normalized_width_height/明确的其他已支持类型）、`depth_kind`（camera_z/ray_distance/absent）、`length_unit`、`metric_status`、`world_frame_description`、`normalization_applied`。其中 `normalization_applied` 是有顺序的变换记录，含参考视图、尺度因子及是否对 depth/translation 一起作用。

如果上游约定确实未知，适配器保存原始输出并标记该字段 `unknown`，核心必须以 `COORDINATE_UNSUPPORTED` 停止该快照的规范化。不能将 unknown 替换成最常见约定后假装成功。原始结果仍可用于排障。

`NativeFramePrediction` 字段：`frame_id`、`depth: ArtifactRef?`、`confidence: ArtifactRef?`、`native_valid_mask: ArtifactRef?`、`K: Matrix3?`、`extrinsic: number[][]?`、`processed_image: ArtifactRef`、`preprocess: ImageTransformChain`、`confidence_semantics: ConfidenceSemantics?`。depth/confidence 尺寸必须分别声明，如不一致须有显式对齐映射。`ConfidenceSemantics` 包含 `source_name`、`higher_is_better: bool`、`value_range: [number,number]?`、`calibration: none|published|locally_fitted`、`calibration_artifact: ArtifactRef?`；未经校准的分数不是概率。

核心 `SnapshotNormalizer.normalize(case, prediction, config) -> BaseSnapshot` 才负责：解析真实约定、转换相机/像素/深度、处理无效值、统一世界框架、建立稳定像素身份并发布规范快照。单帧模型可用本相机原点作为 gauge，但不得为不同单图分别设置 identity 后伪造共享世界。多模型复合基线需要单独命名并记录对齐与融合，不冒称原生 MoGe3 多视图。

## 6. 规范快照：`BaseSnapshot`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `snapshot_id` | `Hash` | 快照内容哈希；持久化后不可变 |
| `case_content_hash` | `Hash` | 输入案例身份 |
| `case_manifest` | `ArtifactRef` | 完整来源案例 |
| `backend_prediction` | `ArtifactRef` | 保留可解释的原生输出与模型信息 |
| `normalizer_version` | `string` | 坐标与数据归一化方法版本 |
| `coordinates` | `CoordinateConvention` | 所有规范数值共同遵守 |
| `frame_order` | `list[Id]` | 稳定 frame_index 的权威顺序 |
| `frames` | `list[SnapshotFrame]` | 按 frame_order 一一对应 |
| `capabilities` | `BackendCapabilities` | 归一化后实际可用能力 |
| `provenance` | `Provenance` | 来源链，正式 hash 投影仅取稳定部分 |

`SnapshotFrame` 字段：`frame_id: Id`、`camera: CameraRecord`、`depth: ArtifactRef`、`valid_mask: ArtifactRef`、`confidence: ArtifactRef?`、`confidence_semantics: ConfidenceSemantics?`、`processed_image: ArtifactRef`。depth 为 float32 `[H_p,W_p]`，单位与快照一致；valid mask 为 bool NPY `[H_p,W_p]`；confidence 为 float32 同 shape，保留原始评分含义；处理图 PNG 的尺寸精确匹配。

规范化后：有效 depth 必须有限且 >0；无效 depth 统一 NaN 且 valid=false；有效点置信值如缺失则 confidence 整体为空，如单像素未定义用 NaN，不因缺分数自动将有效几何变成无效。若 confidence 原始定义与 depth 网格不同，必须经过命名且记录的映射，不能悄悄插值。

第一版不要求为每像素额外保存 XYZ。`domain.snapshot.SnapshotView.iter_points(frame_ids, masks, chunk_size)` 根据 depth、K、T 分块反投影，产生坐标和来源 ID。规范有效点由 valid_mask 决定；置信阈值过滤、下采样和融合属于派生产物，不能改写基础像素身份。

### 稳定点身份

概念类型 `BasePointId = (snapshot_id, frame_id, row, col)`。row 是深度图 v，col 是 u，均从 0 开始；该像素必须在有效掩码中有效。它是基础观测身份，不能假装多个视图看到的同一物理表面天然就是一个 ID。

补丁中共享一次 `base_snapshot_id`，大量 ID 紧凑存为 uint32 `[N,3]`，每行 `(frame_index,row,col)`；frame_index 引用快照 frame_order。数组按这三列字典序排序、去重。显示后的第几个顶点不是 BasePointId；预览下采样必须保存索引映射。首版点云不进行改变身份的显示融合；以后融合需一对多来源映射，不能覆盖此契约。

## 7. 修复输出：`PatchResult`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `patch_id` | `Hash` | 补丁内容身份 |
| `base_snapshot_id` | `Hash` | 唯一允许应用的基底 |
| `selection_content_hash` | `Hash` | 本次目标区域 |
| `method_id` / `method_version` | `Id` / `string` | 首版 `multiview_curve_refiner`，版本明确 |
| `method_config_hash` | `Hash` | 实际参数，不包含 UI 显示设置 |
| `seed` | `int` | 实际修复随机种子，与对应 RunRequest 相同 |
| `outcome` | `accepted_change \| no_supported_change` | 完成计算并不代表改善 |
| `suppressed_points` | `ArtifactRef` | uint32 `[N,3]` 基础点 ID 紧凑数组；可为 `[0,3]` |
| `added_centerlines` | `CenterlineSet?` | 首版主要研究几何 |
| `added_points` | `PointSet?` | 辅助；不能冒充已估计表面 |
| `evidence` | `list[EvidenceRecord]` | 支持、拒绝和不确定性记录 |
| `unresolved` | `list[UnresolvedRegion]` | 有限视差、遮挡、歧义、范围不足等 |
| `provenance` | `Provenance` | 原始输入与方法来源 |

`no_supported_change` 要求 suppression 为空，added 两项均为空。`accepted_change` 至少有真实添加或抑制，且每个改动有对应 evidence。它表示通过算法自身接受规则，不代表经过 GT 证明正确。候选定义始终是“基础有效点 − suppression + 新增几何”；应用补丁不覆盖基底，也不只把新杆叠加在错误连接上。

`CenterlineSet` 字段：`line_ids: list[Id]`、`vertices: ArtifactRef`、`polyline_offsets: ArtifactRef`、`estimated_radii: ArtifactRef?`、`radius_valid_mask: ArtifactRef?`、`radius_provenance: Provenance?`。vertices 为 float32 `[V,3]`，全部有限；offsets 为 int64 `[L+1]`，首项 0、末项 V，每段至少 2 顶点；line_ids 数量 L 且唯一。每条曲线的分段线段长度大于固定数值容差，不能靠重复顶点伪造长度。首版只承诺直杆/分段直杆中心线；交叉的几何线段不自动宣称拓扑连通。

真实半径估计存在时，radii 为 float32 `[V]`，有效位置 >0；无效位置 NaN 且 radius_valid_mask=false；二者必须同时出现，并有估计方法来源。没有半径估计时三个 radius 字段均为空。**显示圆管半径属于 PreviewManifest，不能写入 estimated_radii。** 默认不生成研究用 TubeMesh 或水密表面。

`PointSet` 字段：`point_ids: list[Id]`、`positions: ArtifactRef`、`normals: ArtifactRef?`、`purpose: supporting_samples|independent_geometry`。positions 为 float32 `[N,3]`，全部有限；normals 如有则同 shape、有限且近单位长度，缺法向不能填零伪装。不得把同一 centerline 的稠密采样重复计为额外研究成果。

`UnresolvedRegion` 字段为 `region_ids: list[Id]`、`reason: insufficient_views|insufficient_parallax|occluded|ambiguous_match|unsupported_geometry|numerical_failure|budget_exhausted`、`details: string`。预算耗尽不改成“证据证明不存在细杆”。

### `EvidenceRecord`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `evidence_id` | `Id` | 本补丁内唯一 |
| `target` | `EvidenceTarget` | 新增 line_id/point_id，或 suppression 索引区间 |
| `support` | `list[ViewSupport]` | 多视图观测对应 |
| `checks` | `list[EvidenceCheck]` | 哪些接受/拒绝条件通过 |
| `decision` | `accept \| reject \| unresolved` | 方法判定 |
| `uncertainty_kind` | `heuristic \| calibrated` | 首版通常 heuristic；不能展示成正确率 |
| `calibration_artifact` | `ArtifactRef?` | calibrated 才允许非空，必须来自独立数据 |

`EvidenceTarget` 字段：`kind: centerline|point|suppression|candidate`、`geometry_id: Id?`、`suppression_range: [int,int]?`、`candidate_id: Id?`、`diagnostic_artifact: ArtifactRef?`。centerline/point 只引用已加入几何，suppression 只引用数组半开行号范围；reject/unresolved 使用 candidate_id，若保存被拒候选的诊断数值则放在 diagnostic_artifact，绝不把该文件计入新增几何。只有与 kind 匹配的载荷有值。`ViewSupport` 字段：`frame_id`、`region_ids`、`observed_uv: ArtifactRef`（float32 `[M,2]`，oriented 原图坐标）、`geometry_parameters: ArtifactRef?`（float32 `[M]`，在指定曲线上的累计长度参数）、`visibility: observed|occluded|unknown`、`provenance`。occluded 或 unknown 不能计成可见正证据。

`EvidenceCheck` 字段：`name: Id`、`value: number?`、`threshold: number?`、`unit: pixel|degree|reconstruction_unit|meter|ratio|count`、`comparison: le|ge|boolean`、`passed: bool?`、`note: string`。如为 boolean，value/threshold 为 null，passed 存布尔结果；无法计算时 passed=null 并解释原因。反投影残差、视差角、支持视图数与算法阈值均可记录，但这些不是独立评测指标的替代品。

## 8. 运行请求、状态与记录

### 8.1 `RunRequest`

公共字段：`request_id: Id`、`kind: reconstruct|refine|export_preview|evaluate`、`input: 判别载荷`、`seed: int`、`resources: ResourceRequest`、`requested_by: cli|blender|experiment`、`submitted_at: UtcTime`。run_id 由运行入口生成；submitted_at、request_id 不参与计算缓存键。

| kind | input 的必需字段 | 不变量 |
| --- | --- | --- |
| reconstruct | `case_manifest: ArtifactRef`、`frame_order: list[Id]`、`backend: BackendSpec` | 不携带 GT；预览通过独立 export_preview 运行产生 |
| refine | `base_snapshot: ArtifactRef`、`base_snapshot_id: Hash`、`region_selection: ArtifactRef`、`method_id: Id`、`method_version: string`、`method_config: JsonConfig` | case 从基底追溯；输入 ID 必须与实际快照相同 |
| export_preview | `base_snapshot: ArtifactRef`、`patch_result: ArtifactRef?`、`preview_config: JsonConfig` | 只导出显示派生文件，不重跑算法 |
| evaluate | `evaluation: EvaluationRequest` | 独立分派；不把其引用传入修复器 |

`ResourceRequest` 字段为 `device: cpu|cuda`、`gpu_index: int?`、`max_wall_time_seconds: int?`、`memory_policy: explicit_fail`。不承诺设置显存上限就一定避免 OOM。用户改变分辨率/视图集后提交新请求，不能无记录地重试缩水配置。

### 8.2 `RunStatus`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `run_id` | `Id` | 作业归属 |
| `state` | `queued \| running \| cancel_requested \| cancelled \| failed \| completed` | 只能有一个终态 |
| `stage` | `validate \| wait_resource \| reconstruct \| normalize \| refine \| validate_result \| export_preview \| evaluate \| publish \| finalize` | 当前阶段，不用百分比推断阶段 |
| `sequence` | `int >=0` | 该作业内单调增加，帮助 reader 忽略旧状态 |
| `progress` | `number[0,1]?` | 不可估计则为空；不伪造平滑进度 |
| `message` | `string` | 简短人类可读信息 |
| `updated_at` | `UtcTime` | 心跳/更新时间，不构成进程仍活着的唯一证据 |
| `error` | `ErrorRecord?` | failed 必需；其他状态依场景使用 |
| `result_manifest` | `ArtifactRef?` | completed 时引用已最后发布的 RunRecord；其他状态为空 |

取消命令使用单独小文件 `control/cancel.json`，仅含 schema、run_id、request_id、requested_at、reason；工作进程不能通过读取 UI 场景获知取消。合法状态转移和进程收敛以运行时设计为准。

### 8.3 `RunRecord`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `run_id` / `request` | `Id` / `ArtifactRef` | 唯一运行与冻结请求 |
| `kind` | 与 RunRequest 相同枚举 | 实际运行类型 |
| `state` | `cancelled \| failed \| completed` | 只保存终态 |
| `input_content_hash` | `Hash` | 计算输入与版本身份 |
| `started_at` / `ended_at` | `UtcTime?` / `UtcTime` | 从未启动的排队取消，started_at 可空 |
| `environment` | `EnvironmentRecord` | 核心、后端、宿主实际环境 |
| `stages` | `list[StageRecord]` | 实际成本与阶段结果 |
| `cache` | `CacheRecord` | cache hit/miss/bypass、命中来源和校验结果 |
| `base_snapshot` | `ArtifactRef?` | 已发布规范快照 |
| `patch_result` | `ArtifactRef?` | 有效研究补丁 |
| `preview_manifest` | `ArtifactRef?` | 有效显示包 |
| `evaluation_report` | `ArtifactRef?` | 仅评测运行 |
| `error` | `ErrorRecord?` | 失败或取消原因 |
| `warnings` | `list[string]` | 不隐瞒降级、不可测量项 |
| `reconciled` | `bool` | 是否由恢复检查补写此唯一终态记录 |
| `recovery` | `RecoveryRecord?` | reconciled=true 时必需，否则为空 |

`EnvironmentRecord` 包含 `os`、`python_version`、`core_commit`、`core_lock_hash`、`backend_environments`（backend_id/python/lock/torch/cuda 等实际版本）、`gpu_name`、`driver_version`、`blender_version: string?`、`blender_build_hash: string?`、`determinism_notes`。未测得的硬件/运行值显式 null。

`StageRecord` 包含 `stage`、`started_at`、`ended_at`、`wall_seconds`、`cold_model_load_seconds: number?`、`peak_gpu_allocated_bytes: int?`、`peak_gpu_reserved_bytes: int?`、`peak_process_rss_bytes: int?`、`measurement_method: string`、`outcome`。PyTorch allocator 数值、进程显存与整卡显存不是同一量，不以同一字段混用。`CacheRecord` 包含 `status: hit|miss|bypass`、`key: Hash?`、`source_artifact: ArtifactRef?`、`validation: full_hash|none`；命中仍需文件身份验证。

`RecoveryRecord` 字段为 `reconciled_at: UtcTime`、`reason_code: Id`、`original_processes_confirmed_exited: bool`、`known_exit_code: int?`、`termination_reason: string?`、`evidence_log: ArtifactRef?`。只有确认原任务及后代已经结束且未发布终态时，才能补写失败终态；不能改写已有不可变终态。未知退出原因显式为空，不从取消文件推断已完成有序取消。

终态状态文件和最终 RunRecord 的发布顺序须一致：先发布产物，再发布 RunRecord，最后写 completed 状态。失败记录可以引用排障数据，但不能把临时半成品标成有效 snapshot/patch/preview。

## 9. Blender 唯一展示入口：`PreviewManifest`

预览由核心验证并导出，Blender reader 不自己组合 raw depth 或执行补丁算法。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `preview_id` | `Hash` | 显示包内容身份，与研究身份分开 |
| `base_snapshot_id` | `Hash` | 来源基底 |
| `patch_id` | `Hash?` | 有候选才存在 |
| `source_run_id` | `Id` | 可追溯运行 |
| `core_to_blender` | `Matrix4` | 已应用的展示变换 B，仅允许正尺度 × proper rotation + translation；不回写研究数值 |
| `coordinate_space` | 固定 `blender_world` | 所有预览位置与相机已经应用 B，导入器不得再次应用 |
| `display_length_unit` | `string` | 如 `reconstruction_unit`，不得未知冒充米 |
| `display_scale` | `number >0` | 说明 B 中已包含的展示倍率，不再乘第二次 |
| `base` | `PreviewVersion` | 完整基础显示版本 |
| `candidate` | `PreviewVersion?` | 完整应用补丁后的显示版本 |
| `cameras` | `list[PreviewCamera]` | 已完成宿主边界转换的显示相机 |
| `display_settings` | `DisplaySettings` | 抽样和线粗等，仅影响显示 |
| `validated_by` | `Provenance` | 核心导出与验证版本；不是密码学签名 |

`PreviewVersion` 字段：`version_id: Id`、`role: base|candidate`、`blocks: list[PreviewBlock]`、`bounds: [Vec3,Vec3]`、`point_count: int`、`segment_count: int`、`warnings: list[string]`。每份版本表达其完整显示状态，candidate 包含抽样后保留的基础点及新增显示几何；抽样预算与采样规则在 base/candidate 间一致，suppression 必须先应用到完整基础数据，再按固定 ID 采样规则导出。

`PreviewBlock` 字段为 `block_id: Id`、`kind: base_points|added_points|line_segments`、`artifact: ArtifactRef`、`layout_version: "1"`、`record_count: int>=0`、`record_stride: int`、`source_mapping: ArtifactRef`、`mapping_layout: u32x3-le-v1|point-id-json-v1|segment-id-json-v1`。每个块有固定数量上限，导入器逐块处理；几何与 mapping 的条目数必须严格对应。

只使用以下 packed little-endian 二进制布局，不带隐式 header 或 padding；元数据由 JSON 提供。文件长度严格等于 record_count × record_stride，标准库 `struct` 可读取，不依赖 Blender 内的 NumPy、通用 PLY 或 NPZ reader。

| kind | 每条记录 | stride | source_mapping |
| --- | --- | --- | --- |
| base_points | position `float32 ×3` + RGB `uint8 ×3` + flags `uint8` | 16 bytes | 独立 `.bin`，每行 `uint32 ×3`，12 bytes，表示 frame_index,row,col |
| added_points | position `float32 ×3` + RGB `uint8 ×3` + flags `uint8` | 16 bytes | JSON 字符串数组，逐行对应 PointSet.point_ids |
| line_segments | start/end `float32 ×6` + RGB `uint8 ×3` + flags `uint8` | 28 bytes | JSON 对象数组，每行 `{line_id, edge_index}`，edge_index 为该 polyline 内从 0 起的线段号 |

首版 flags 必须为 0，RGB 是诊断颜色而非恢复的材质。位置必须有限；Blender 插件可把 segments 构造成显示曲线和有粗细的圆管，这属于宿主派生显示，不是核心输出的真实 TubeMesh。原始 polyline 没有显式 junction，因此不承诺原生拓扑连接正确；评测若使用图连接，须标为从固定 CurveReadout 推断的图。

`DisplaySettings` 字段：`max_base_points: int>0`、`max_added_points: int>=0`、`max_segments: int>0`、`max_total_records: int>0`、`max_total_geometry_bytes: int>0`、`max_records_per_block: int>0`、`sampling_method: string`、`sampling_seed: int`、`tube_radius_mode: "fixed_display"`、`fixed_display_radius: number>0`、`radius_unit: "blender_world_unit"`、`change_color_rgb: [int,int,int]`、`base_color_mode: string`。预算同时约束 base/candidate 两份总几何及映射，不能只限制基础点而让新增线段无限增长；省略任何显示几何都必须记入 warnings 与显示数量，研究产物不截断。该半径在已变换的 Blender 展示坐标中解释；不再乘 display_scale。首版以固定显示粗细统一显示，不声称这是物理半径；独立半径估计如存在仍只保存在研究产物中。

`PreviewCamera` 字段：`frame_id`、`matrix_world: Matrix4`、`K: Matrix3`、`image_size_wh`、`image: ArtifactRef`、`camera_fit_policy`。matrix_world 按 `B × inverse(T_cw) × diag(1,-1,-1,1)` 构造；Blender 相机向前 -Z、向上 +Y。宿主 lens/sensor/shift 的配置需和所导出的 K 对齐；无法精确表达的投影必须拒绝相机导入或明确降级为图像平面，不把外观近似当标定成功。

## 10. 评测接口

`EvaluationRequest` 顶层字段固定为：`evaluation_id: Id`、`suite_ref: ArtifactRef`、`base_snapshot_ref: ArtifactRef`、`base_snapshot_id: Hash`、`patch_result_ref: ArtifactRef?`、`compared_run_refs: list[ArtifactRef]`、`ground_truth_ref: ArtifactRef?`、`protocol_ref: ArtifactRef`。请求内容哈希绑定具体基底、协议与切分。GT 只允许评测进程读取；普通修复器不接收此 DTO。

`EvaluationReport` 顶层字段为：`evaluation_id`、`base_snapshot_id`、`protocol_hash: Hash`、`case_results_ref: ArtifactRef`、`aggregate_ref: ArtifactRef`、`alignment_ref: ArtifactRef`、`provenance_ref: ArtifactRef`、`report_ref: ArtifactRef`、`warnings: list[string]`。定量值、不可计算原因、全局对齐与统计口径放在引用的评测文件中；只有二维辅助量的真实案例不伪装成已证明三维精度。

评测器读取研究数值文件与协议规定的几何读出，不读取 Blender 派生圆管或预览抽样点。基础点云与候选中心线存在表示差异，不能直接用“新增点更多”证明恢复率提高；统一读出、公平比较、对齐与误补全指标以独立评测设计为准。

## 11. 内容身份、缓存与发布

### 11.1 哈希投影

所有内容对象使用固定函数 `content_hash(document_type, schema_version, semantic_payload)`：先验证模式，再按字段名排序的 canonical JSON 编码 UTF-8（无空白、无 NaN、字符串不做隐式 Unicode 改写、-0 规范为 0），前置明确的 document_type 和 schema_version 域标识，计算 SHA-256。实施时冻结 canonicalizer 版本并提供黄金样例；不把普通 `json.dumps` 的偶然默认设置当协议。

`semantic_payload` 使用每种 DTO 的显式白名单，不是“删掉几个已知字段，其他都算”；嵌套 Provenance 和 parent_artifacts 同样递归使用稳定字段白名单，不能从子对象重新带入路径或时间。照片、数组、普通配置文件的引用参与身份的是字节 hash、字节长度、格式、shape/dtype/meaning/unit；不包含本机 store 根、bundle_id 或 path。对已定义内容身份的 JSON 文档引用，如 CaseManifest、BackendPredictionManifest、BaseSnapshot、PatchResult，内容投影使用解析并复算验证过的语义内容 ID，不能使用会受路径/时间变化影响的 manifest 文件字节 hash。ArtifactRef.sha256 仍用于该文件的完整性检查；两种哈希不能混用。重新编码图片/NPY 会改变字节 hash，首版接受此结果，不实现跨编码语义去重。迁移存储位置只改本机 bindings 时无需重写任何文档；确需重写引用的迁移产生新文件引用，但不改变其已验证的研究内容身份。

| 对象 | 纳入身份 | 排除身份 |
| --- | --- | --- |
| Case | 有序 frame IDs、原文件与 oriented 文件 hash、方向/颜色转换、前景 mask、相机/尺度来源、静态输入属性 | 自身 hash、描述文字、revision、created_at、磁盘位置、photo_preview及显示映射 |
| Selection | case hash、最终 region 几何/来源、有效传播方法版本及参数 | 自身 hash、操作时间、UI actor 名、会话 |
| Backend prediction | 稳定请求身份、实际 backend/模型/环境配置、原生数组 hash、实际 conventions/preprocess | 自身 ID、run ID、计时、缓存、日志 |
| Snapshot | case hash、backend prediction 内容身份、normalizer 版本、coordinates、frame_order、规范相机和数组 hash | 自身 snapshot_id、创建时间、存放路径、缓存命中信息 |
| Patch | base_snapshot_id、选择/方法/参数、seed 来源、suppression、新增几何、有效证据与未解决原因 | 自身 patch_id、run ID、耗时、显示设置 |
| Preview | 来源 snapshot/patch、完整显示文件 hash、B、显示设置 | 自身 preview_id、source_run_id、创建时间、宿主当前选中状态 |

ID 计算必须避免循环：先写数组并获得其字节 hash，再以这些 hash 计算 snapshot_id；数组内不重复存 snapshot_id。BasePointId 的 snapshot 部分由所属快照/补丁元数据提供。补丁哈希可包含固定的 base_snapshot_id，不构成循环。将 bundle 存入摘要目录之后才写最终引用，内容投影忽略这些存储位置。

### 11.2 缓存不是运行记录

重建缓存键是计算请求，不等同于最终 snapshot_id：输入照片与顺序、已知相机/尺度、预处理、模型权重与代码、环境锁、精度、推理参数及 seed 均纳入；只有算法输入相同才允许重用。修复缓存键额外包含 base_snapshot_id、区域内容、方法版本、参数和种子。预览缓存额外包含 display 配置，预览抽样不能污染研究缓存。

缓存索引单独保存 `key → ArtifactRef`，可重建；`cache_hit`、last_accessed、运行时间、run_id、磁盘路径不参与内容 hash。命中不能重写不可变快照的 provenance 或计时。新的 RunRecord 记录此次命中与验证成本，原始生成成本仍从来源运行追溯。

一个请求可能受非确定 GPU 运算影响而产生不同 snapshot_id；缓存采用已发布的一个明确结果，不承诺相同请求总是逐位同结果。正式实验报告确定性设置和重复运行差异。

### 11.3 逻辑目录与原子发布

```text
data/cases/<case_id>/
  manifest.json
  originals/<frame_id>.<ext>
  oriented/<frame_id>.png
  photo-previews/<frame_id>.png
  selections/<selection_id>.json
cache/snapshots/<snapshot_digest>/
  manifest.json
  frames/<frame_id>/depth.npy
  frames/<frame_id>/valid.npy
  frames/<frame_id>/confidence.npy          # 仅模型提供时
runs/<run_id>/
  request.json
  status.json                             # 可原子替换的轻量状态
  control/cancel.json                     # 有取消请求时
  logs/core.log
  logs/backend.stdout.log
  logs/backend.stderr.log
  work/                                   # 未发布，reader 不导入
  artifacts/backend/prediction.json
  artifacts/backend/frames/<frame_id>/...
  artifacts/result/patch.json
  artifacts/result/suppressed.npy
  artifacts/result/centerlines/vertices.npy
  artifacts/result/centerlines/offsets.npy
  artifacts/preview/manifest.json
  artifacts/preview/base/...
  artifacts/preview/candidate/...
  run-record.json                         # 终态记录，产物完成后发布
data/evaluation/<evaluation_id>/
  request.json
  protocol.json
  cases.json
  aggregate.json
  alignment.json
  report.md
```

上述是逻辑默认布局，不要求用户照片一开始已经在这里；输入准备阶段复制/导入到受管案例包并校验内容。原照片不被修改。案例有修订时创建新的不可变案例包 ID 或版本包，不能覆盖旧 manifest 后让旧 run 指向改变的内容。

每个文件先写同一文件系统下的临时文件，flush/close、验证长度与 hash 后再原子 rename/replace 发布。大型包先完成所有文件，最后发布 manifest；跨盘 copy 不能被当作原子 rename。状态文件可以原子替换，研究产物发布后禁止覆盖。存在 run 目录并不意味着成功；reader 只接受完整最终清单以及逐个验证过的引用。

## 12. 最小线格式示例

以下是完整 `ArtifactRef` 与 `RunStatus` 示例；重复字母摘要仅用于展示线格式，实际 reader 必须核验对应真实文件，不能将示例摘要当成可用产物。

```json
{
  "store": "run",
  "bundle_id": "run_0001",
  "path": "artifacts/result/centerlines/vertices.npy",
  "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "byte_size": 176,
  "media_type": "application/x-npy",
  "array": {
    "dtype": "float32",
    "shape": [4, 3],
    "order": "C",
    "meaning": "world_centerline_vertices",
    "unit": "reconstruction_unit"
  },
  "members": null
}
```

```json
{
  "schema_version": "1.0.0",
  "document_type": "RunStatus",
  "run_id": "run_0001",
  "state": "running",
  "stage": "refine",
  "sequence": 7,
  "progress": null,
  "message": "正在检查候选细杆的多视图支持",
  "updated_at": "2026-09-13T08:00:00Z",
  "error": null,
  "result_manifest": null
}
```

## 13. 契约实施时必须通过的检查

1. 支持版本接受、未知版本拒绝；冻结 canonical JSON 黄金样例；移动文件不改变内容身份，改像素、相机、参数或 suppression 必须改变对应身份。
2. 图片方向、半像素 resize/crop、非线性去畸变、CV/Blender 坐标使用独立标记点检查；错误 K、pose 方向与未声明单位必须拒绝。
3. hash 不符、路径越界、reparse 跳转、NPY object dtype、shape 不符、有效位置 NaN/负深度和不完整 manifest 均拒绝；使用真实坏输入验证，不只照着代码断言字段存在。
4. 同一 snapshot 的 BasePointId 在抽样、显示、保存重开后仍对应原像素；补丁绑错基底、重复 ID、无效像素 ID 或越界 ID 均拒绝。
5. `no_supported_change` 不携带修改；抑制与新增的完整候选可再现；显示圆管半径改变不改变研究 patch hash，也不改变独立评测数值。
6. 后端缺共享外参时，多图修复明确失败；保存 unknown conventions 的原生包仍不得发布假 canonical 快照。缓存命中只产生新的运行记录，不污染旧产物。

这些检查验证的是协议可信和系统一致性。算法能否恢复细杆、何时误补全，仍由独立实验回答。
