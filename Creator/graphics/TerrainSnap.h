#pragma once
#include <glm/glm.hpp>
#include <unordered_map>
#include <string>

class Scene;
class Terrain;

struct SnapRules {
    float yEpsilon = 0.02f;   // 通用抬升，避免 Z-fight
    bool  onlyRaise = false;  // 仅抬高（不向下压）
    bool  ignoreWater = true; // 水体不贴地

    // 按类别的附加偏移（单位：米），可为负数（flora 允许轻微“入土”）
    // key: "building"/"prop"/"flora"/"fauna"/"terrain"/"water"
    std::unordered_map<std::string, float> categoryYOffset;
};

/// 遍历 Scene 内所有实例，对每个实例的“脚底”(AABB 最小 y)与地形高度对齐：
/// worldMinY(model) → terrain.heightAtWorld(x,z) + yEpsilon (+ per-category offset)
void SnapSceneToTerrain(Scene& scene, const Terrain& terrain, const SnapRules& rules = {});

