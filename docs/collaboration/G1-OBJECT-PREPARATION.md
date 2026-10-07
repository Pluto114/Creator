# G1 不同装配输入准备（2026-10-06）

状态：新代码与 10 张 RGB 已完成并冻结，生成校验通过；尚未标定/运行 DA3/方法/共同评分，不代表已有新对象改善。Codex 子助手 g1_objects 负责数据与机制测试，主代理负责核心算法。

两种对象为新合成椅背/椅座装配与 A 字支架；不是实拍、独立采集或盲测。复用同一个渲染器与已知尺寸双深度标定板。方法仍限定 RGB 预声明单杆 ROI，不扩展为整物体多杆修复。

## 已实现入口与冻结边界

`configs/g1_object_pilot_v1.json` 固定新 run `g1-object-scenes-v1-20261006`、chair01/aframe01 × 5 views = 10 RGB，继承旧 640×480、49 mm、5.8 m 半径、五方位/高度、48 samples、灯光和 CAD。parent prepared/config/CAD 均绑定 SHA；chair target [0,0,-.75]→[0,0,.8]、radius .012；A-frame target [-.05,0,-.8]→[.05,0,.85]、radius .009。完整座面/两侧框/横框与 A 型前后腿/撑/底杆是不同装配上下文，参数仅在生成协议与 GT。

`scripts/blender_g1_object_pack.py` 是薄 wrapper：验证固定屏幕 guide 后复用旧 fixture.setup、height.position、identity.setup_case/main；缺少 guide、替换来源、目标 world guide、未知/重复 case 均拒绝，不允许走旧 project_guide 兜底。

`scripts/prepare_g1_object_pack.py` 验证固定配置和父源码 SHA，显式继承旧 fixture source inventory 并加 4 个新源码/config/test 及已核对的传递依赖，不 glob 正在开发的新算法。before-render source snapshot 和 generation-freeze 先写；任何已有 run/input/truth/evaluation 路径拒绝；逐 case 独立 Blender；全帧独立射线和 Z/range 检查；完成清单最后发布。冻结后上述对应源码不得修改，失败另起版本。

已由主代理阅核并授权执行（现已完成；同 run 再执行会拒绝覆盖）：

```powershell
. .\scripts\Enter-CreatorEnvironment.ps1
.\backends\da3\.venv\Scripts\python.exe -B scripts\prepare_g1_object_pack.py
```

## 复用依赖与下游合同

渲染：blender_g1_object_pack → blender_rod_fixture_pack → blender_rod_height_pack → blender_rod_reference_pack → blender_rod_identity_pack → blender_export_thin_pack。已有 setup_case 支持 target + 任意方向 distractors + 一块可定位 cube，无新几何实现。

准备：prepare_rod_reference 的 source_snapshot/locations/raster_center_truth/validate_render_manifest、prepare_rod_heldout_views.array_checks、thin_pack_gt.MeshRays/check_blender_rays。GT 操作只属生成阶段，不进入正常重建；代码查看与此次机制测试未读 eval_gt。

正常 manifest 严格构造允许字段：顶层 run_id/cases/declared_cad/fixture_artwork_sha256/scope；case 只有 case_id/frames；frame 只有 view_id/rgb/rgb_sha256/size_wh/guide_xyxy/guide_source。不放 source_asset_id、object_group_id、标签、构件坐标、半径、生成相机或表面标签。case_id 是匹配用 ID，不是输入几何。两个 asset/group 的合成身份仅在生成协议与 GT 中报告。

标定可复用 creator_eval.fixture_calibration 的 detect_markers/calibrate_case/POLICY，逐 case 单独拟合 5 RGB，不池化对象、不读生成相机。新标定 runner 必须参数化 scene ID，不能调用路径绑定旧 scene 的 run_fixture_calibration.checked。

杆法可复用 extract_fixture_narrow_evidence/reconstruct_fixture_narrow 及旧完整 method（SHA b33ce471de3fbe6b7387d78494c7675108a68793fa9d68711a5eb6a1f4c3072e）。旧单目标近竖直 strip、minimum_y_span=150 px、所有观测/拒绝保持；不能由装配新增横杆推断方法已支持全物体。

DA3/实际补丁沿用 camera_inputs → smoke_da3.run(large,504,align_to_input_ext_scale=true) → exported_cameras → 导出相机下重算 RGB 杆 → materialize/source_roundtrip → 不可变完整快照与 write_patch → 保存启用/撤回并 reopen。新 runner 需移除旧 r01/r02/r03 路径、case 数和 18 行硬编码；共同两尺度 reader 由主代理选择，完整基底、朴素对照及实际候选同域比较，不沿用旧 abstention reader 默认为合格。

正常方法继续拒绝 protocol、render_request、generation-checks、rendered-rgb staging、eval_gt、evaluation 与物理结果；生成源码可只核验字节 SHA，不能把其中 target 几何作为方法输入。完整正常输出及独立 pre 先于 separate GT score。重复运行用新 ID、相同输入/源码/配置，不覆盖首轮失败。

固定屏幕 guide 不从 endpoints 投影、无逐帧人工标注；装配设计、渲染准备与 RGB 验看耗时另记，不能宣称人工总成本为 0。两对象均保留；若任一标定/方法失败不旋转、缩放、改 guide 后覆盖旧 run。

## 本助手动作记录

- 只读核查旧 renderer、fixture/height/reference prepare、标定、finite/narrow、DA3/point patch 与 G1 原始定义；未读取 GT。
- 新建上述 4 文件及本 MD；未修改旧冻结源、未运行 Blender/DA3。
- 首轮 9 项机制测试在 NumPy 1.26 与 NumPy 2 环境均通过；静态检查首次仅 import 顺序问题，已修正通过。
- 依赖核查发现旧 inventory 缺 run_rod_identity_blender.py 与 reconstruction 两层 __init__.py，均显式补入新 inventory；未扩大到正在改动的 naive 模块。
- 增加已有部分运行不得覆盖测试后，共 10 项双环境通过（0.025/0.024 秒），Ruff 通过。
- 主代理阅核后授权 prepare；首尝试在创建任何 run/input 前被历史源码校验拒绝：rod_observations.py 的 09/27 SHA 为 ee7e4856…，09/29 已冻结的 narrow 新实现与当前源均为 941b712f…。其余父源全相同，未 render、未生成输出。
- 获主代理批准，只为该文件加 source_migrations：原 parent snapshot 按旧 SHA 仍完整验证；新 narrow prepared SHA 为 83d3b26133bb8705bf0af257de3d127336030c6c8238ab30076913d46a991df1，新 snapshot/current 按新 SHA 验证；非许可文件或其他变化拒绝。机制测试增至 13 项，两 NumPy 环境均通过（0.032/0.031 秒），Ruff 通过。未修改旧源。
- 再次正式执行 prepare exit 0，生成 chair01/aframe01 各 5 张 RGB；冻结 67 源文件与显式迁移清单。generation-checks=passed，11,598 独立射线、surface_id_mismatches=0；最大投影误差 4.402144315918122e-05 px，最大 Z/range 校验误差 7.663731569351739e-07 m。只做生成真值校验，未读取方法物理评分。
- 正常 manifest SHA：bd56da4a289c78280d641d2893eb2e548b82e4758a46324a00b908a4dfaa722c。prepared 与 generation-checks 位于 `.runtime/experiments/g1-object-scenes-v1-20261006/`；RGB 位于 `data/inputs/g1-object-scenes-v1-20261006/{chair01,aframe01}/view_00.png` 至 `view_04.png`。已人工查看两对象 view_02 RGB，确认不同完整装配与中央细杆可见，未据此筛掉其他帧或改 guide。
- 已将 prepared/checks SHA、全部 42 项正常输入 SHA（manifest、CAD、10 RGB、30 artwork）、逐层字段允许列表与 10 个固定屏幕 guide 复核、67 源 live/snapshot 终检、首次阻断与单文件迁移记录保存至 `docs/experiments/results/2026-10-06-g1-object-inputs.json`。prepared SHA d437619e2fd66412343b7f92c92ef7d96714e7998df3b801d6e8f366f763b2ed；checks SHA 79ccbb3aafa4d8ab9900ad9434e595048f49c2a218a7b5ffa99c23912457c038。source mismatch=0；报告明确 generation_only、reconstruction_run=false、method_scoring_run=false、g1_passed=false。

## 后续正常 RGB 核验（同日）

主代理/另一助手完成 `g1-object-rgb-v1-20261006` 后，本助手仅在 `run_g1_object_rgb.block_truth` 启用下调用 checked/verified_inference 完整复核并摘要：22 源、229 收据、2 对象 × 5 验证相机、4 个方法输出均完整，error_object_count=0，原 infer 耗时 99.2903373 秒。chair 两法均 accepted 1；A-frame baseline accepted 1、cylinder_support rejected 0。方法未单独计时，报告保留 null+说明，不将共享 case 时间伪装成每法耗时。

A-frame 正常拒绝链直接来自记录：初始圆柱边缘门要求至少 4 视图、每侧至少 90% 边缘在 1 px 容差；仅 view_00/view_01 通过，view_02/03/04 双侧比例分别 .814/.814、.855/.855、.711/.791。因此仅留 2 视图，stored assignment 不 eligible，继而 no_stored_assignment_passes_cylinder_screen → association_not_accepted。该屏仅检查已存 best/support competitors，不代表逐一检查所有 17 个 unique hypotheses；未做 subset search。没有 GT，不能据此称目标漏检、baseline 正确或收益成立。

完整正常摘要保存于 `docs/experiments/results/2026-10-06-g1-object-rgb-normal.json`，绑定 prepared/inference/input/method 配置文件 SHA、22 源 SHA、229 收据映射的 canonical SHA、两案各视图标定 p95 与拒绝门直接数据。该报告 normal_only=true、gt_read=false、method_physical_scoring_performed=false、g1_passed=false；未修改冻结源、未读取新 GT、未运行 DA3 或评分。

## 旧 fixture 单轴共同读出对照（同日，完成）

主代理实现单轴核心及 adapter，本助手仅新增 `run_fixture_single_axis_readout.py`、`fixture_single_axis_readout_v1.json`、`test_fixture_single_axis_readout.py`。以旧 naive 60 正常行逐一配对真实完整 bundles，五臂、三案、两尺度、两次 reader 重放；不重拟合、重建或修改补丁。原 mask/曲线采样等 7 个计数字段与 native/control/call 完全一致才允许发布。normal 全部验证后另进程评分；原生新增几何单列重算，不由读出 mask 隐藏假桥。

12 项 runner 机制测试双环境通过（0.076/0.081 秒），Ruff 通过；显式 235 源包含主核/adapter 与 DeepSeek 6 项独立测试（SHA b599ab10d8875f59be5f8f21f336d76a2a634e2931067d4755a2d2753797b084），没有新增扫描 glob。主代理完整阅核后授权运行；`fixture-single-axis-readout-v1-20261006` prepare/infer/evaluate/post 全部 exit 0，9794 pre 收据、60 fresh rows、post 精确重放通过。冻结后未再修改源码。两次重复仅是 reader 独立进程重放，不能称端到端重建重复或新对象验证。

结果完整保存在 `docs/experiments/results/2026-10-06-fixture-single-axis-readout.json` 及 audit：无 normal/scoring 错误，30/30 reader 物理投影重复一致，60/60 原生补丁几何及物理重算不变。正常 infer 墙钟 259.5276005 秒，其中 60 次 reader 调用合计 1.0595192 秒，其余含独立进程、来源校验与输入读取等；不是完整重建耗时。

关键失败不隐藏：r03 纯 base 从空恢复 R=47.0%/68.9%、P=100%；但该全局轴拟合也使旧完整候选回退。r01 baseline/cylinder 原 R=P=100% 变为 R=59.4%/60.0%、P=59.68%/60.28%，边界误差 44.88/42.37 mm；r02 细尺度 baseline R/P=85.09%/85.58%、cylinder=87.64%/88.02%，粗尺度两者均 R=P=0%，仍输出两段而位置偏移。r03 两候选仍 R=P=100%，边界最大 6.74/9.75 mm；其 TLS/RANSAC 新读出约 R=98.3%、P=100%，边界约 59.8 mm。base 三案在两尺度分别 5/4、3/3、4/4 段；r01 base R=39.3%/44.4%，r02 base R=0%。所有 shared 保护缺口误长为 0，但不能据此宣称其他位置/错误增长无误，原 native 指标仍单列。

结论：此版证明共享 reader 不应系统丢掉纯 base，但目前 reader 噪声偏轴与尺度敏感性使候选明显回退，不能直接作为合格共同读出取代旧主表，更不构成 G1 通过。全部失败版本与结果已保留。
