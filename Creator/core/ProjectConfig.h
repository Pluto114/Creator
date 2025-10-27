// core/ProjectConfig.h
#pragma once
#include <string>

struct ProjectConfig {
    // 资产根目录（PlacementsLoader 的 opts.assetsRoot）
    std::string assetsRoot = "assets/models";
    // 资产注册表（可为空）
    std::string registryPath = "";
    // 占位模型路径（可为空）
    std::string placeholderModel = "assets/placeholders/cube.glb";
    // 启动时要加载的 placements.json 绝对或相对路径（相对工作目录）
    std::string placementsPath = "";
};

// 从 json 文件读取配置；读取失败返回 false（并保留 out 的默认值）
bool LoadProjectConfig(const std::string& jsonPath, ProjectConfig& out);

// 把配置写回 json；写失败返回 false
bool SaveProjectConfig(const std::string& jsonPath, const ProjectConfig& cfg);

