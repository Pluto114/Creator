// core/ProjectConfig.cpp
#include "ProjectConfig.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

static void ToJson(json& j, const ProjectConfig& c) {
    j = json{
            {"assetsRoot",       c.assetsRoot},
            {"registryPath",     c.registryPath},
            {"placeholderModel", c.placeholderModel},
            {"placementsPath",   c.placementsPath}
    };
}
static void FromJson(const json& j, ProjectConfig& c) {
    if (j.contains("assetsRoot"))        c.assetsRoot       = j["assetsRoot"].get<std::string>();
    if (j.contains("registryPath"))      c.registryPath     = j["registryPath"].get<std::string>();
    if (j.contains("placeholderModel"))  c.placeholderModel = j["placeholderModel"].get<std::string>();
    if (j.contains("placementsPath"))    c.placementsPath   = j["placementsPath"].get<std::string>();
}

bool LoadProjectConfig(const std::string& jsonPath, ProjectConfig& out) {
    std::ifstream ifs(jsonPath);
    if (!ifs) {
        std::cerr << "[ProjectConfig] Cannot open " << jsonPath << " (using defaults)\n";
        return false;
    }
    try {
        json j; ifs >> j;
        FromJson(j, out);
        std::cout << "[ProjectConfig] Loaded from " << jsonPath << "\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ProjectConfig] Parse error: " << e.what() << " (using defaults)\n";
        return false;
    }
}

bool SaveProjectConfig(const std::string& jsonPath, const ProjectConfig& cfg) {
    try {
        json j; ToJson(j, cfg);
        std::ofstream ofs(jsonPath);
        if (!ofs) return false;
        ofs << j.dump(2);
        return true;
    } catch (...) { return false; }
}

