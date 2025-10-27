#pragma once
#include "D:\Creator\Creator\graphics\TerrainSnap.h"

// 在 ImGui 中绘制“地形工具”面板，编辑贴地规则、开关自动贴地。
// 参数：autoSnap 开关（引用），rules（引用，面板可修改其字段）
void DrawTerrainToolsPanel(bool& autoSnap, SnapRules& rules);

