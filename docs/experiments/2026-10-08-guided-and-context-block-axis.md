# 2026-10-08：guide方向分块与局部unknown（开发回归）

从b90759f继续，其CI37742762777已成功。本轮不修改已有freeze，新增guided-block及context-block两个研究版本；仍不替换生产读取器。全部是已见合成开发数据，reader-only重复不等于DA3端到端独立重复，G1仍未过。

## 决策依据与不变项

原guide是粗搜索条，不是前景身份。若改为只保留覆盖guide的唯一RGB edge pair，五案原base支持点569/555/120/271/13245变为72/0/80/197/1，r02/Aframe被清空；这条方案只作normal诊断，未用于正式评分。原r02粗base竞争轴相对guide仅0.195/0.998度，单方向筛选无法排除噪声拟合。正常诊断保存62份输入收据及SHA，未读GT。

两版均用原guide与同坐标系估计相机的图像直线平面SVD生成分块方向，位置完全不参与；原RGB union、完整base、曲线采样、尺度、origin、native补丁都保持原样。每案各952560基础点。两尺度0.0025/0.005倍原相机跨度，五臂base/TLS/RANSAC/baseline/cylinder，两个独立reader子进程。

guided-block按方向每4v分块，两个横向投影使用0.5v bins。局部可靠bin必须来自至少2个不同轴向voxel，全局至少3个；可靠带空隙>v视为分裂，连续3块且两侧移动≤2v判持续双轨。短双轨或不足证据也不默认单杆。最终轴来自实际world-voxel均衡3D TLS，不投guide；支持半径max(v,代表点横距中位数+3MAD)，实际轴向占据AND原RGB，无gap closing。

这些直方图只检验有限分辨率的两个横向投影，不是一般单峰、语义身份、表面或拓扑合格证明。配置中的强歧义/表面拒绝措辞仅指保留既有判据，不表示所有表面均已验证可拒绝。

## 第一版完整失败（guided-block）

| 套件 | 冻结/完整性 | 新reader实绩 |
| --- | --- | --- |
| 旧fixture | 262源、9848pre；60行，30/30重复，60/60native不变 | 全60空，12base+48candidate；34非持续split、26持续多轨；R0/Pnull；相对independent新增48空、12物理不变 |
| 新对象 | 365源、1300pre；80共同+10native，40/40重复 | 新40行36空；24非持续split、10持续多轨、2全局带不足；仅chair粗TLS/RANSAC各1段、R=P100，端点21.742/21.625mm |

两套四阶段均passed、方法及评分错误0，只证明完整记录与复算，不证明算法成功。对象原40 global正常图除计时及physical、10 native逐字段同10/7。fixture normal120.261s，60次reader合计0.993s；对象normal81.166s，新40次reader2.102s。共享机器耗时不当作独立速度基准。

## 第二版：局部未知不能抹掉整个对象

context-block保留所有分裂/持续门，不以改阈值增加输出。持续双轨或全局band不足仍拒绝；只有非持续的局部分裂变成固定guide/origin块上的unknown。先从拟合及观测支持中排除该块所有点，再重算voxel代表/TLS/MAD；最后axis samples再次显式排除同一unknown块，防止占据半径跨回去。原完整GT/评分域完全不裁剪，未知部分仍算漏检。保存unknown块、点及采样数，有局部输出标partially_resolved。

独立斜杆短分叉例验证输出两侧真实斜线、不跨未知块；全unknown明确支持不足。新增测试最初硬编码31未知点与linspace端点表示不一致，冻结前改为按该测试实际坐标独立计算点数，未知块与几何断言不变。v1所有失败保持冻结。

### 新对象完整结果（context-block）

375源/1320pre、62parent输出收据，80共同+10native，40/40重复一致，0方法/评分错误；40原global与10native继续精确不变。

| chair方法 | 细：段数；R/P；长度误差mm | 粗：段数；R/P；长度误差mm |
| --- | --- | --- |
| base | 4；55.35/100；−854.022 | 3；93.81/100；−205.978 |
| TLS | 4；56.65/100；−846.033 | 1；100/100；−30.012 |
| RANSAC | 4；52.13/100；−910.026 | 1；100/100；−29.987 |
| baseline | 4；61.03/100；−782.034 | 2；93.81/100；−173.994 |
| cylinder | 同baseline | 同baseline |

Aframe五臂两尺度全部空：10持续多轨、2全局band不足、8非唯一主轴。新40行20空，chair20行有输出；16行相对guided从空恢复，24物理不变，无新增空。相对上一轮independent则16不变、12恢复、4新增空（Aframe baseline）、8非空结果改变，不能只报相对本轮失败版的改善。

chair细/粗base仍弱于旧global的R97.42/99.74，恢复的片段不是完整杆；除粗TLS/RANSAC外均边界数量不匹配，端点误差不可测。normal74.334s，新40次reader2.050s、最大0.135s，global40次0.702s。新两对象没有真gap，gap0不作为缺口合格证据；native的Aframe TLS/RANSAC误补仍完整保留。

### 旧fixture完整结果（context-block）

272源/9868pre，60行完整，30/30重复、60/60native及COUNT_KEYS相同，四阶段post passed，0方法/评分错误。26空，34非空但都是局部unknown碎段；相对guided恢复34行、26物理不变，无新增空。相对上一轮independent仅4物理不变、8空转非空、22非空退空；8行R上升、40行R下降，明显不能升级主流程。

下表R/P为百分数，N为段数；两repeat一致，空结果P不可测而非100。

| case/arm | 细：R/P；N | 粗：R/P；N |
| --- | --- | --- |
| r01 base | 13.2/37.5；5 | 0/—；0 |
| r01 TLS | 0/—；0 | 0/—；0 |
| r01 RANSAC | 0/0；6 | 0/—；0 |
| r01 baseline/cylinder | 0/—；0 | 0/—；0 |
| r02 base | 0/—；0 | 32/45.714；4 |
| r02 TLS | 0/—；0 | 0/0；3 |
| r02 RANSAC | 15.636/100；2 | 32.242/45.714；4 |
| r02 baseline | 0/—；0 | 50.182/75；4 |
| r02 cylinder | 0/—；0 | 71.636/100；4 |
| r03 base | 0/—；0 | 69.7/100；4 |
| r03 TLS | 86.5/100；3 | 93.6/100；2 |
| r03 RANSAC | 86.6/100；3 | 93.2/100；2 |
| r03 baseline/cylinder | 86.5/100；3 | 95.3/100；2 |

base长度分别短1425.210/534.447/800.257mm；仅r02细RANSAC边界数量匹配，端点误差793.863mm，其余null，不当作0。guarded-gap误补均0，但其他位置仍有误输出。normal109.301s，60次reader合计0.947s，平均0.015786s、最大0.030803s。

## 可复现结果与动作日志

四个新入口为scripts/run_fixture_guided_block_axis_readout.py、run_g1_object_guided_block_readout.py及对应context版本，各自prepare/infer/evaluate/post独占run，禁止覆盖重跑。公开结果及audit见[本轮结果目录](results/)，正常产物/源码快照保存在.runtime/experiments的同名run。

DeepSeek任务guided-block-contracts-001交付6机制测试与本人日志，Codex仅修E402及lambda风格并实际运行；18核心测试双环境0.120/0.218s。context的6项复用是Codex适配，不是第二次助手任务。固定D:/deepseek/collaboration/WORK_LOG.md及work-log.json现17任务。

三个Python包离线隔离构建通过：3.667/1.748/1.955s，183源SHA不变，六产物与命令见.runtime/validation/packages-20261008-002/outcome.json。对象guided命令日志最初写到根.validation，已验证精确路径后将6份文件按原字节归档到.runtime/validation/g1-object-guided-block-20261008-001；原命令中的历史路径不改，无数据删除。

context最终17主+6复用共23核心双NumPy通过（0.109/0.192s），runner29另双环境通过。本轮两版共99项新增/复用契约，含重复机制，不是99个独立对象。最终一次全量1772/1772通过，0失败/错误/跳过，282.540s；core pytest16 tests+15 subtests通过；Ruff/370源/三CLI通过，652份source/config/workflow SHA前后相同。证据.runtime/validation/numpy126-full-20261008-002/outcome.json；收集SHA22d84251fea4fcc32bb2da5d46f3c17b0c79f9d041b884b6488958f15064cdb6，源清单SHA8fed87a16301d393db422ec2c3b8950d1d55db273704bb6504bd4b02ac37bb41。

| 公开结果 | 同目录audit | public SHA256 |
| --- | --- | --- |
| [fixture guided](results/2026-10-08-fixture-guided-block-axis-readout.json) | [audit](results/2026-10-08-fixture-guided-block-axis-readout-audit.json) | 2cea76413b538742439befdae58044abf923bba96c73633b1b1e0c8dbd2c5247 |
| [object guided](results/2026-10-08-g1-object-guided-block-readout.json) | [audit](results/2026-10-08-g1-object-guided-block-readout-audit.json) | e6ce0b405b1690409f14c1a6730d755c6bd883b865caa671e127d4202a29319b |
| [fixture context](results/2026-10-08-fixture-context-block-axis-readout.json) | [audit](results/2026-10-08-fixture-context-block-axis-readout-audit.json) | 08a00322e1c71bb3c50fe076611ba3f6da7290514998f13dc22748f18bb8f4a4 |
| [object context](results/2026-10-08-g1-object-context-block-readout.json) | [audit](results/2026-10-08-g1-object-context-block-readout-audit.json) | 587726689d92f36517f37384ba68845588c5b19701f3715e7f4ed5f4487223b5 |

## G1结论与下一主线

**G1:55%（维持）**，六项仍6+8+15+10+8+8。相对本轮guided坏版本恢复50条重复记录，不等于相对上一轮总体进步；对independent仍有26条非空退空及大量碎段/位置回退。主流程不替换，不宣称完整恢复、公平读取已过或总体优于基础。

下一轮不要继续按空行数加读取器阈值。核心问题是RGB候选union中的多结构归属与真实DA3噪声混合；先用原图/现有edge-pair证据核验同一前景链的跨视图归属，保留所有竞争解释。原guide不能冒充精确前景anchor；若确需新增人工选择，独立冻结并公开计入输入/人工成本，所有对照共用。局部unknown保留机制，但固定完整评价域；只有公平读取与不同对象净收益可核验后再跑真正DA3+补丁端到端独立重复。没有新增实拍，不重复催问、不捏造。
