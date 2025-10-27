#include "Environment.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

static bool fileExists(const std::string& p){
    std::error_code ec; return fs::exists(p, ec);
}

static void from_json(const json& j, EnvironmentSettings& s){
    if (j.contains("skyboxFolder"))  s.skyboxFolder = j["skyboxFolder"].get<std::string>();
    if (j.contains("skyboxColor"))   { auto v=j["skyboxColor"]; s.skyboxColor = { v[0], v[1], v[2] }; }
    if (j.contains("ambientColor"))  { auto v=j["ambientColor"]; s.ambientColor= { v[0], v[1], v[2] }; }
    if (j.contains("sunDirection"))  { auto v=j["sunDirection"]; s.sunDirection= glm::normalize(glm::vec3(v[0],v[1],v[2])); }
    if (j.contains("sunColor"))      { auto v=j["sunColor"];     s.sunColor    = { v[0], v[1], v[2] }; }
    if (j.contains("sunIntensity"))  s.sunIntensity = j["sunIntensity"].get<float>();
    if (j.contains("fogEnabled"))    s.fogEnabled   = j["fogEnabled"].get<bool>();
    if (j.contains("fogColor"))      { auto v=j["fogColor"];     s.fogColor    = { v[0], v[1], v[2] }; }
    if (j.contains("fogDensity"))    s.fogDensity   = j["fogDensity"].get<float>();
    if (j.contains("terrainPreset")) s.terrainPreset= j["terrainPreset"].get<std::string>();
}

static json to_json(const EnvironmentSettings& s){
    return json{
        {"skyboxFolder",  s.skyboxFolder},
        {"skyboxColor",   {s.skyboxColor.r, s.skyboxColor.g, s.skyboxColor.b}},
        {"ambientColor",  {s.ambientColor.r, s.ambientColor.g, s.ambientColor.b}},
        {"sunDirection",  {s.sunDirection.x, s.sunDirection.y, s.sunDirection.z}},
        {"sunColor",      {s.sunColor.r, s.sunColor.g, s.sunColor.b}},
        {"sunIntensity",  s.sunIntensity},
        {"fogEnabled",    s.fogEnabled},
        {"fogColor",      {s.fogColor.r, s.fogColor.g, s.fogColor.b}},
        {"fogDensity",    s.fogDensity},
        {"terrainPreset", s.terrainPreset}
    };
}

bool Environment::loadFromJson(const std::string& jsonPath) {
    useDefaults();
    std::ifstream ifs(jsonPath);
    if (!ifs) {
        std::cerr << "[Environment] Cannot open " << jsonPath << " (using defaults)\n";
    } else {
        try {
            json j; ifs >> j;
            from_json(j, cfg);
            std::cout << "[Environment] Loaded from " << jsonPath << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[Environment] Parse error: " << e.what() << " (using defaults)\n";
        }
    }
    return applySettings();
}

bool Environment::saveToJson(const std::string& jsonPath) const {
    try {
        std::ofstream ofs(jsonPath);
        if (!ofs) return false;
        ofs << to_json(cfg).dump(2);
        return true;
    } catch (...) { return false; }
}

void Environment::useDefaults() {
    cfg = EnvironmentSettings{}; // 默认值
}

void Environment::applyToShader(Shader& shader) {
    shader.setVec3("uAmbientColor", cfg.ambientColor);
    shader.setVec3("uSunDirection", cfg.sunDirection);
    shader.setVec3("uSunColor",     cfg.sunColor * cfg.sunIntensity);
    shader.setInt ("uFogEnabled",   cfg.fogEnabled ? 1 : 0);
    shader.setVec3("uFogColor",     cfg.fogColor);
    shader.setFloat("uFogDensity",  cfg.fogDensity);
}

bool Environment::applySettings() {
    if (!cfg.skyboxFolder.empty() && fileExists(cfg.skyboxFolder)) {
        return sky.loadFromFolder(cfg.skyboxFolder);
    } else {
        sky.setSolidColor(cfg.skyboxColor);
        return true;
    }
}

void Environment::drawSkybox(const glm::mat4& view, const glm::mat4& projection) {
    sky.draw(view, projection);
}

