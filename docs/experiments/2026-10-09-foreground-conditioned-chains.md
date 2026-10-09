# 显式前景选择、稀疏笔画与必选视图同链支持

日期：2026-10-09。基点 dd97dbbdddb921c78dc1a4a91d4b408a4b805d19。**G1:55%，未通过，生产主流程不切换。** 两版四 run 全部四阶段完成；Aframe 从全空恢复部分正确短段，但尚未恢复完整杆，必选视图还损伤其他 base/候选。以下是已见合成开发数据上的输入/支持干预，不是盲测、实拍或新 DA3 端到端重复。

## 决策与实现

停止将匿名全链 union 当作已知目标。先用两个独立 RGB 视图的显式前景标记筛固定候选链；单点不能限制走向，再用分离行的稀疏标记要求**同一图像候选**支持每个标记；最后检查投票是否绕过了选择视图。

| 读取策略 | 支持域 | 额外输入 |
| --- | --- | --- |
| raw / single_axis | 原 RGB bands 任意至少 3 视图 | 不使用新标记，记录共享清单 |
| all-chain / chain_axis | 一条固定赋值内至少 3 视图，所有赋值 OR | 不使用新标记，记录共享清单 |
| anchored_chain_axis | 两视图单点条件下的正支持赋值，仍至少 3 视图 | 每案 2 点 |
| stroke_chain_axis | 每视图所有标记共同支持同一候选，仍至少 3 视图 | 每案 6 点 |
| required_stroke_axis | 同 stroke 赋值内，所有标注视图必须支持，且总票至少 3 | 与 stroke 完全相同 |

完整候选池、pair-plane 搜索、1.5px 匹配/1.2px row 成员门、至少 4 视图链、预算及原单轴 TLS/物理门都未调。输出仍拟合实际 3D 点/曲线采样，不投回图像轴。所有臂保留相同完整输入、冻结尺度、原始 RGB 票、native 几何和完整 GT 域。新 mask 是正观测条件，不是场景“没有杆”的证据；所有策略 `target_identity_confirmed=false`。

点框的三态：全部覆盖行有某一真实 band 包含整个框才 supported；全部覆盖行有观测且每一 band 都不与框相交才 contradicted；其余 unresolved。未知链留在审计、不贡献正支持，最终质量损失仍在完整 GT 域计入。稀疏点之间没有插值。预算不完整时整体回退原 raw，必选视图版显式报告没有应用 required 约束。

## 新输入来源与成本

两个输入版本均来自原始 RGB 视觉检查和指定行像素检查，明确为助手注释，非用户点击、非盲标；没有从 GT、native 杆、3D 提议或评分选坐标。历史聚合开发结果已见，不声称整个研发过程盲化。所有点在观测带检查前冻结，检查后零修订；raw band 全包支持不等于正确目标身份。

| 输入 | 新增点 / 总点 | 新增原图查看 | 新增 RGB 行精读 | 冻结后 band 检查 |
| --- | ---: | ---: | ---: | ---: |
| anchors-v1 | 10 / 10 | 10 | 10 | 2 次 |
| strokes-v1 | 20 / 30 | 10 | 20 | 1 次 |

标注助手累计 30 个不同点、20 次原图查看、30 次行像素精读；主代理另复核 Aframe 原图 1 次，未改点。第一版原始支持 10/10，第二版 30/30；各点不确定度 ±0.5px。没有测得可靠人工耗时，不能把记录时间差伪装为标注耗时。笔画版额外 20 点必须计成本，不能把其提升全部归于纯算法。

- [点选 manifest](../../data/inputs/rgb-foreground-anchors-v1-20261009/manifest.json)：`b87cb3f91aba295648d59e9cdf807301e19c4c0a01b6af2ac565858073cc7c05`。
- 点选 audit：`3365eb91d30c1e4fe40055f509af2bc8ff12449ef82e024f59fb25021800891e`。
- [笔画 manifest](../../data/inputs/rgb-foreground-strokes-v1-20261009/manifest.json)：`f18dc283fe5b4cecb6aecec201368b97bab446d3fc0b7b21572eaecf50ce2963`。
- 笔画 audit：`7452fd5c5c865acb9a6965a863491587ec920c1bf11e3cead4e26746d23a28f8`。

## 正常诊断：为什么增加必选视图

只使用原图观测、估计相机、完整 base；不使用本轮 GT 或 native 几何。独立可复跑脚本与 JSON 保存在 `.runtime/validation/stroke-normal-core-review-20261009-001/`，37 项显式源/输入 SHA 前后不变，耗时 34.679s（不是完整冻结运行的性能基准）。

| 案例 | 全链数 | 点选链数 | 笔画链数 | 点选 / 笔画 base 点 | 必选视图后点数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| r01 | 32 | 17 | 14 | 379 / 379 | 189 |
| r02 | 18 | 11 | 8 | 390 / 390 | 192 |
| r03 | 1 | 1 | 1 | 103 / 103 | 73 |
| chair01 | 5 | 3 | 3 | 221 / 221 | 122 |
| aframe01 | 28774 | 1058 | 68 | 7257 / 2068 | 109 |

Aframe 单点版的两尺度次特征值比 .617504/.531134，仍拒绝；笔画版 .140814/.126327，读出一段，但不证明几何正确。2068 个点中 1959 个在原 3/5 同链票策略下绕过至少一个标注视图。因此新增独立 bitset `all-required AND minimum-three`；它在**每一完整赋值内部**计算后才 OR，不能先跨链 union 两个必选视图。

## 正式运行协议

点选版 `fixture-anchored-chain-readout-v1-20261009` / `g1-object-anchored-chain-readout-v1-20261009` 分别 180/120 正常行；笔画版 `fixture-stroke-chain-readout-v1-20261009` / `g1-object-stroke-chain-readout-v1-20261009` 分别 240/160。每版包含五臂（base、TLS、RANSAC、baseline、cylinder）、两冻结尺度、两个独立读取进程；700 行包含大量相同控制和重复，不是 700 个独立案例。

所有四套完整正常记录和 SHA 核验后才允许本轮评价；原 raw/all-chain 控制要求逐字段精确同 10/8。点版与笔画版各自保留源、配置、输入、association、正常记录和评分；不会覆盖点版失败。三个旧案仍同一资产家族，新两案仍合成装配。复用原 DA3 快照和补丁，不能称作重新运行 DA3。

入口为 `scripts/run_{fixture,g1_object}_{anchored,stroke}_chain_readout.py prepare|infer|evaluate|post`。执行前加载 `scripts/Enter-CreatorEnvironment.ps1`，使用已有 DA3 Python。现有阶段禁止覆盖重跑；后续方法改动必须新源/配置/run ID。

### 正常输出与评分屏障

| run | 源数 / pre 收据 | 正常行 | 完整重复对 | 正常执行秒 |
| --- | ---: | ---: | ---: | ---: |
| fixture-anchored | 294 / 10325 | 180 | 90 | 170.434 |
| object-anchored | 397 / 1838 | 120 | 60 | 266.396 |
| fixture-stroke | 310 / 10357 | 240 | 120 | 184.523 |
| object-stroke | 413 / 1870 | 160 | 80 | 215.498 |

四次正式推理各自完成校验后，另在禁止 GT 的进程核验全体 700 行、350 重复对、全部输入收据和旧控制；全部通过、0 方法错误。主代理核验屏障后才授权评价。屏障 `.runtime/validation/stroke-chain-runners-20261009-001/four-normal-barrier.json`，SHA `cea79dfe707259739d3a36791aa550e583e831853e5ec683cd1eb0fdd16b5675`。以上时间包括正常 reader/context 运行，与回归共享 CPU，不作算法加速基准；既有 DA3、标注、prepare/验证成本不包含在该列，不能称端到端耗时。

### 实现验证和工作日志

一次全量 1947/1947、0 失败/错误/跳过，core 16 tests＋15 subtests，Ruff、384 源/三个 CLI 通过；703 项源/config/build-input/4 个注释 JSON 的集合和 SHA 前后不变。三包离线构建生成 6 产物，177 个 wheel Python 模块逐字节匹配冻结源码，覆盖本轮全部 6 个新模块。真实收据在 `.runtime/validation/{numpy126-full-20261009-001,packages-20261009-001}`。

核心归属、三态规格修正、失败及分工见[当天开发日志](../journal/2026-10-09.md)。DeepSeek 任务 `anchored-chain-contracts-001` 提交 6 个三态契约及本人 work-log，实际测试由 Codex 执行；固定 `D:/deepseek/collaboration/WORK_LOG.md` / `work-log.json` 共 19 项任务。源码与回归通过不替代 G1 几何结论。

## 完整评分结论

四 run 的 evaluate/post 全部完成，700 正常/评价行、350 物理重复对一致；全部方法/评分错误为零。两个 object 版本各保留 10 条 native 记录，复用同一组 native 对照，不是 20 个独立方法。公开结果与审计：

另一次独立结果审计核对13158个唯一文件收据并重算255组、4794684个base/curve/output点查询，`required⊆stroke⊆chain⊆raw` 无违例，正式计数逐行相同；全部R升降表独立复算一致。收据 `.runtime/validation/foreground-results-review-20261009-001/outcome.json`。审计工具首轮误把拟合路径/SHA/计时当物理差异，原记录保留；修正工具比较投影后通过，正式源和结果未改。

- 点选：[fixture 结果](results/2026-10-09-fixture-anchored-chain-readout.json)、[audit](results/2026-10-09-fixture-anchored-chain-readout-audit.json)；[object 结果](results/2026-10-09-g1-object-anchored-chain-readout.json)、[audit](results/2026-10-09-g1-object-anchored-chain-readout-audit.json)。
- 笔画：[fixture 结果](results/2026-10-09-fixture-stroke-chain-readout.json)、[audit](results/2026-10-09-fixture-stroke-chain-readout-audit.json)；[object 结果](results/2026-10-09-g1-object-stroke-chain-readout.json)、[audit](results/2026-10-09-g1-object-stroke-chain-readout-audit.json)。
- 全部 public/audit SHA 和四阶段日志索引：`.runtime/validation/stroke-chain-runners-20261009-001/four-run-delivery.json`。

### 全部 base，不挑成功例

每格“细 / 粗”，R 为百分比；点选和 stroke 的前四案 base 物理结果相同。除 Aframe 外，点选/stroke/required 的 base P 均为100%，但碎段和缺失长度仍是失败。Aframe 点选为空、P=null；stroke 非空但 P=0%；required 只有很短一段、P=100%，不代表完整恢复。

| base | raw R | all-chain R | 点选 R | stroke R | required R | stroke 段数 | required 段数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r01 | 39.3 / 44.4 | 55 / 69.7 | 80.2 / 88.7 | 80.2 / 88.7 | 55.5 / 75.4 | 6 / 4 | 5 / 4 |
| r02 | 0 / 0 | 95.636 / 100 | 95.636 / 100 | 95.636 / 100 | 68.485 / 96 | 7 / 3 | 5 / 5 |
| r03 | 47 / 68.9 | 43.5 / 53 | 43.5 / 53 | 43.5 / 53 | 34.2 / 52.9 | 4 / 3 | 3 / 4 |
| chair01 | 97.419 / 99.742 | 91.742 / 95.871 | 91.742 / 95.871 | 91.742 / 95.871 | 72 / 76.903 | 6 / 2 | 5 / 3 |
| aframe01 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 7.739 / 12.455 | 1 / 1 | 1 / 1 |

required base 缺少长度（毫米，细/粗）：r01 1104.798/678.546，r02 706.603/200.755，r03 1465.334/1138.155，chair 646.187/499.713，Aframe 1573.128/1494.021。前四案边界数不匹配，不能给虚构端点误差；Aframe 虽单段，边界最大误差仍1513.636/1434.786mm。缺口误填零不抵消这些完整性失败。

### Aframe 全五臂

原 raw/all-chain/点选均全部空。stroke 两尺度五臂全部非空，但除了 baseline 的少量覆盖，其他四臂 R=P=0，说明“非空”确实是错误几何。required 拦截绕过视图的支持后有部分正确覆盖，仍不能接受。

| 方法臂 | stroke R / P / 段数（细；粗） | required R / P / 段数（细；粗） |
| --- | --- | --- |
| base | 0 / 0 / 1；0 / 0 / 1 | 7.739 / 100 / 1；12.455 / 100 / 1 |
| TLS | 0 / 0 / 1；0 / 0 / 1 | 7.739 / 100 / 1；12.455 / 100 / 1 |
| RANSAC | 0 / 0 / 1；0 / 0 / 1 | 7.739 / 100 / 1；12.455 / 100 / 1 |
| baseline | 1.935 / 4.825 / 3；14.994 / 47.917 / 2 | 44.256 / 77.679 / 2；43.652 / 78.182 / 2 |
| cylinder（原生拒绝，保留base） | 0 / 0 / 1；0 / 0 / 1 | 7.739 / 100 / 1；12.455 / 100 / 1 |

required baseline 总长892.667/872.447mm，仍少760.361/780.580mm且两段，边界数不匹配。Aframe native TLS/RANSAC 仍是错误长杆，R=P=0、长度1790.420/1079.930mm，不能被共同读取器裁掉后隐藏；native baseline 原有R100/P97.880、边界59.356mm的结果也原样保留。这说明当前公共几何拟合仍会损伤本来正确的新增杆，不能把提升至44%当读取资格通过。

### 其他候选与全套升降

required 的旧 baseline/cylinder 三案两尺度 R=P100，正确段数1/2/1、边界≤14.873mm；但同一规则下 r01 RANSAC 细/粗仅77.8/87%、各2段，r02细TLS仅R82.667/P82.843。r03 TLS/RANSAC分别R98.3/98、P100，但base仅34.2/52.9，资格问题仍未解决。chair 的四候选由 stroke 的R=P100降至R99.226–99.355/P100、边界35.825–36.134mm；不是零回退。

以下统计包含两个进程重复，只是完整矩阵行数，不当独立样本。各行 `R↑ / ↓ / =`：

| 新策略 | 组 / 行数 | 对 raw | 对 all-chain | 空结果 |
| --- | --- | --- | --- | ---: |
| 点选 | fixture / 60 | 40 / 8 / 12 | 20 / 0 / 40 | 0 |
| 点选 | object / 40 | 0 / 4 / 36 | 0 / 0 / 40 | 20 |
| stroke | fixture / 60 | 40 / 8 / 12 | 20 / 0 / 40 | 0 |
| stroke | object / 40 | 4 / 4 / 32 | 4 / 0 / 36 | 0 |
| required | fixture / 60 | 40 / 8 / 12 | 22 / 14 / 24 | 0 |
| required | object / 40 | 20 / 20 / 0 | 20 / 20 / 0 | 0 |

新对象没有真 gap，因此 gap=0不能证明缺口安全；旧 fixture 的原 guarded-gap 误填检查完整保留。人工成本增加、多个 base 回退、native 好杆经共同拟合受损均计入结论，不用较前一失败版本的进步掩盖与原控制的退化。

## G1 与下一步

**G1:55%，六项仍 6+8+15+10+8+8。** 显式输入来源、同链必选视图和完整消融已落地，但不同对象可重复收益/共同读取公平性仍未关闭；不因测试或实验数增长加分，不切换生产主流程。

下一核心任务是**支持归属与实际几何拟合分离**：已有明确前景标记仍把多种深度/方向的实际点合在一次全局 TLS 中，且硬性逐点视图交集损失 r03/chair 稀疏覆盖。先用正常记录检查同链局部实际点的几何竞争、base 与曲线混合后的轴偏移，以及 missing-view 与反证的差异；采用同输入、同阈值的分链/稳健真实几何读取新版本，不再单纯收紧掩码，不把标记三角化成评测输出，不按 native/GT 挑轴。方法仍须新冻结全矩阵；资格通过后再做真正 DA3＋补丁端到端重复。
