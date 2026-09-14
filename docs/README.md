# 文档索引
最新工作状态：[裁剪/真实相机/直线对照](experiments/2026-09-14-crop-oracle-line-controls.md)。9次新增模型推理完成，并在13份预测上比较了普通/鲁棒拟合与RGB多视图交会。[下一版算法设计](decisions/0004-evidence-guided-line-recovery.md)已由结果确定；独立对象改善、可靠拒绝和完整补丁仍待实现。重现见[对照说明](thin-pack-controls.md)，此前基线和数据包说明继续保留。

日期：2026-09-13。适用根目录：`D:/Creator-newage`。

开发过程与判断记录：[2026-09-14 日志](journal/2026-09-14.md)。

本组架构文档是首版实现规格。依赖锁定与本机验证进度见 [环境说明](environment.md)；正式几何算法及研究效果仍需验证。

项目骨架已建立；已实现内容与运行命令见 [开发说明](development.md) 及 [根 README](../README.md)。下方详细架构仍是目标设计，未实现模块不会假装已完成。

| 顺序 | 文档 | 解决的问题 |
| --- | --- | --- |
| 环境 | [安装与空间管理](environment.md) | 独立依赖、模型下载、缓存、健康检查与 Blender 启动 |
| 实测 | [环境验证](environment-verification.md) | 本机版本、实际模型输出、耗时与显存及其限制 |
| 1 | [系统设计](architecture/01-system-design.md) | 产品范围、五层职责、模块依赖、目录、公开入口与实现顺序 |
| 2 | [数据契约](architecture/02-data-contracts.md) | 每种输入输出的字段、数组、坐标、身份、版本和文件协议 |
| 3 | [重建与局部修复](architecture/03-reconstruction-and-refinement.md) | 适配器、几何服务、候选算法、类与方法、拒绝条件和补丁语义 |
| 4 | [任务运行与 Blender](architecture/04-runtime-and-blender.md) | 进程、缓存、取消、恢复、插件界面、导入和工作副本 |
| 5 | [实验与交付](architecture/05-evaluation-and-delivery.md) | 评测模块、指标、对照、公平条件、关键测试与三个月验收 |
| 决策 | [ADR 0002](decisions/0002-reconstruction-stack.md) | 当前技术栈与本机事实 |
| 决策 | [ADR 0003](decisions/0003-newage-architecture.md) | 详细架构的固定边界和可替换部分 |
| 决策 | [ADR 0004](decisions/0004-evidence-guided-line-recovery.md) | 由裁剪、相机与直线对照选择下一版恢复方案 |
| 历史 | [ADR 0001，已废止](decisions/0001-technology-stack.md) | 旧场景生成路线，仅保留决策历史 |

## 如何解释“确定”

- **必须遵守**：坐标定义、不可变快照、版本协议、真值隔离、结果不覆盖人工编辑、取消只影响本任务。
- **首版选择**：Python 核心、DA3 适配器、中心线表示、文件进程协议、照片 ROI、直杆候选修复方法。
- **待验证**：依赖组合、运行预算、对应与拟合参数、算法收益、可靠的半径估计。实验可以推翻这些选择。
- **延后**：任意网格修补、材质恢复、自动完整拓扑、模型训练平台、云服务、多机调度、自写渲染器。

每个文档中的 Python 风格签名都是接口说明，不是实现。`T`、`Sequence[T]`、`ArrayRef` 等表示约定类型；具体 DTO 以数据契约为准。普通内部计算使用函数或小型数据记录，不为每一步建立工厂和继承树。

## 变更与冲突处理

数据字段、单位和身份以 [02 数据契约](architecture/02-data-contracts.md) 为准；任务终态及发布顺序以 [04](architecture/04-runtime-and-blender.md) 为准；质量与实验判定以 [05](architecture/05-evaluation-and-delivery.md) 为准。架构层与包路径以 [01](architecture/01-system-design.md) 为准。

发生矛盾应在实现前修改文档和相应测试，不在代码里偷偷添加另一套约定。模式变更须同步类型定义、导出的 JSON Schema、协议样例及读取器兼容检查。
