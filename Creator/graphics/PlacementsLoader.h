#pragma once
#include <string>

class Scene;

/// 运行时选项（与 PlacementsLoader.cpp 一一对应）
struct PlacementLoadOptions {
    // 作为“最后一跳”的根目录（可被环境变量 CREATOR_ASSETS_ROOT 覆盖）
    std::string assetsRoot = "assets/models";
    // 资产注册表（可留空 "" 表示不用）
    // 支持两种格式：
    //  1) { "map": { "id": "path.glb", ... } }
    //  2) { "id": "path.glb", ... }  扁平映射
    std::string registryPath = "";
    // 占位模型（当找不到真实模型时）
    std::string placeholderModel = "assets/placeholders/cube.glb";
    // 载入前是否清空场景
    bool clearSceneBeforeLoad = true;
};

/**
 * 从 placements.json 读取并实例化到 Scene。
 * 路径解析优先级：
 *  1) placements.models[i].asset_path
 *  2) registry.json 中的 map[src_object] 或扁平映射
 *  3) 环境变量 CREATOR_ASSETS_ROOT / opts.assetsRoot + 推测的 <src>.glb/.gltf/.obj
 *  4) opts.placeholderModel
 */
bool LoadPlacementsIntoScene(const std::string& placementsJsonPath,
                             Scene& scene,
                             const PlacementLoadOptions& opts = {});

