#include "TerrainToolsPanel.h"
#include <imgui.h>

static const char* kCats[] = { "building", "prop", "flora", "fauna", "terrain", "water" };

void DrawTerrainToolsPanel(bool& autoSnap, SnapRules& rules) {
    if (ImGui::Begin("Terrain Tools")) {
        ImGui::Checkbox("自动贴地 (每帧)", &autoSnap);
        ImGui::SeparatorText("通用规则");
        ImGui::SliderFloat("通用抬升 yEpsilon (m)", &rules.yEpsilon, -0.05f, 0.1f, "%.3f");
        ImGui::Checkbox("仅抬高 (不向下压)", &rules.onlyRaise);
        ImGui::Checkbox("忽略水体", &rules.ignoreWater);

        ImGui::SeparatorText("按类别偏移 (m)");
        for (const char* c : kCats) {
            float v = 0.0f;
            auto it = rules.categoryYOffset.find(c);
            if (it != rules.categoryYOffset.end()) v = it->second;
            if (ImGui::SliderFloat(c, &v, -0.1f, 0.1f, "%.3f")) {
                rules.categoryYOffset[c] = v;
            }
        }

        if (ImGui::Button("重置偏移为推荐值")) {
            rules.categoryYOffset["building"] = 0.02f;
            rules.categoryYOffset["prop"]     = 0.01f;
            rules.categoryYOffset["flora"]    = -0.02f;
            rules.categoryYOffset["fauna"]    = 0.00f;
            rules.categoryYOffset["terrain"]  = 0.00f;
            rules.categoryYOffset["water"]    = 0.00f;
        }
    }
    ImGui::End();
}

