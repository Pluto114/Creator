#pragma once
#include <string>
#include <glm/glm.hpp>
#include "Shader.h"
#include "Skybox.h"

// 环境参数（默认可用）
struct EnvironmentSettings {
    std::string skyboxFolder = "";
    glm::vec3   skyboxColor  = glm::vec3(0.05f, 0.08f, 0.12f);

    glm::vec3   ambientColor = glm::vec3(0.15f);
    glm::vec3   sunDirection = glm::normalize(glm::vec3(-0.3f, -1.0f, -0.2f));
    glm::vec3   sunColor     = glm::vec3(1.0f);
    float       sunIntensity = 1.0f;

    bool        fogEnabled   = false;
    glm::vec3   fogColor     = glm::vec3(0.5f, 0.6f, 0.7f);
    float       fogDensity   = 0.02f;

    std::string terrainPreset = "";
};

class Environment {
public:
    bool loadFromJson(const std::string& jsonPath); // 读配置（读不到走默认）
    bool saveToJson(const std::string& jsonPath) const; // ★ 新增：保存配置
    void useDefaults();
    void applyToShader(Shader& shader);
    void drawSkybox(const glm::mat4& view, const glm::mat4& projection);

    // ★ 新增：把当前 settings 应用到运行时（主要是天空盒）
    bool applySettings();

    EnvironmentSettings& settings() { return cfg; }
    const EnvironmentSettings& settings() const { return cfg; }  // ★ 新增：允许 const 对象访问


private:
    EnvironmentSettings cfg;
    Skybox sky;
};

