# 点快照补丁试行协议与复现

适用目录`D:/Creator-newage`。实现是`reconstruction/src/creator_recon/domain/point_patch.py`，协议版本`0.1.0`。它用于已有点数组的不可变快照和有限中心线补丁，尚未替代架构保留的正式深度文件契约。

## 数据与操作

| 产物 | 内容 |
| --- | --- |
| `base/manifest.json`及数组 | `<f8`点坐标，`<u4`来源ID，原始图像/预测SHA、帧表、坐标系、单位、点有效性策略 |
| `patch/manifest.json`及数组 | 有限线段、待抑制来源ID、绑定快照、方法版本/配置SHA/种子、选择SHA、接受证据与未决原因 |
| `view-enabled.json` | 同级base/patch目录引用、精确内容ID、应用状态、自身内容哈希 |
| `view-withdrawn.json` | 同一补丁的撤回状态；读取时仍检查补丁有效性 |

`write_snapshot`和`write_patch`只创建新目录。`compose(base, patch, enabled=True)`装配原点减抑制ID再加有限段；`enabled=False`恢复精确原点。`write_candidate_view`只创建新收据，`open_candidate_view`按已保存状态装配。保存视图使用同级简单目录名，拒绝路径逃逸、软链/重解析点和内容替换。哈希保证可追踪和意外改动检测，不是来源作者的密码学签名。

点ID按帧、行、列排序且唯一。删除引用不得越界或不属于绑定快照。新增与抑制的每项必须恰好有一条接受证据；拒绝/未决证据不能授权改动。当前证据为来源产物SHA与视图引用，真实性由冻结的运行链核查，单独一个SHA不能证明几何正确。

完整Schema：[快照与补丁](../schemas/pilot/point-patch-0.1.schema.json)、[保存视图](../schemas/pilot/candidate-view-0.1.schema.json)。Schema处理结构；Python读取器继续检查数组哈希、类型、形状、ID、绑定及证据覆盖。单包中途失败会留下未发布目录供排查，读取器不把没有最终manifest的目录当成完成。

## 本机复现

先设置D盘环境，不重新安装模型：

```powershell
Set-Location D:\Creator-newage
. .\scripts\Enter-CreatorEnvironment.ps1
```

读取已有结果并重新核验，不覆盖冻结推理：

```powershell
& .\backends\da3\.venv\Scripts\python.exe scripts\check_point_patch_runs.py
& .\backends\da3\.venv\Scripts\python.exe scripts\report_g1_point_patch.py
& .\reconstruction\.venv\Scripts\python.exe -m pytest reconstruction\tests
& .\backends\da3\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_common_readout*.py' -v
```

前两条只读冻结模型/运行结果，重新生成派生审计与图。第一条需要已安装的`jsonschema`，报告绘图使用已安装的Matplotlib。Git仅保存汇总，不能在无本地输入的全新克隆上直接运行历史回放；原始模型输出仍需按既有配对包和DA3命令产生。

新运行需复制相应配置，改`run_id`并换公共输出路径；保留其他参数才算相同协议复跑。例如以下新名字尚未占用时可运行首版：

```powershell
Copy-Item configs\g1_point_patch_v1r2.json .runtime\g1-point-patch-replay.json
$runConfig = Get-Content .runtime\g1-point-patch-replay.json -Raw | ConvertFrom-Json
$runConfig.run_id = 'g1-point-patch-replay-01'
[System.IO.File]::WriteAllText('D:\Creator-newage\.runtime\g1-point-patch-replay.json', ($runConfig | ConvertTo-Json -Depth 20), [System.Text.UTF8Encoding]::new($false))
& .\backends\da3\.venv\Scripts\python.exe scripts\run_g1_point_patch.py prepare --config .runtime\g1-point-patch-replay.json
& .\backends\da3\.venv\Scripts\python.exe scripts\run_g1_point_patch.py infer --run-id g1-point-patch-replay-01
& .\backends\da3\.venv\Scripts\python.exe scripts\run_g1_point_patch.py evaluate --config .runtime\g1-point-patch-replay.json --share-json .runtime\g1-point-patch-replay-summary.json
```

第二版入口为`run_g1_point_patch_components.py`，对应`g1_point_patch_components_v1.json`，相同三阶段。当前源码默认首版r2与分量v1均匹配冻结值；r1须使用该运行的`source_snapshot`，不能直接拿已修正源码重评旧ID。

`prepare`验证历史预测/候选/RGB哈希并冻结输入与源码；`infer`不读取评价GT；`evaluate`才核验GT索引并做统一对齐和评分。源码变化、读取器规则变化必须用新运行，保留双方完整结果。

## 明确未完成

两版共同读取器仍不合格，见[首轮报告](experiments/2026-09-22-point-patch-and-common-readout.md)。当天[追加工作](experiments/2026-09-22-readout-split-and-estimated-cameras.md)已实现第三版读取器，并把前景身份接入4份新DA3估计相机快照与18个补丁，但读取资格和几何效果仍未通过。本轮没有实际点抑制算法、任务取消/恢复、正式DTO迁移或Blender导入/撤回。不要把“补丁能保存”写成“普通照片已经修好了”。
