#include "EnvironmentPanel.h"

#include <imgui.h>
#include "tinyfiledialogs.h"
#include <string>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "D:\Creator\Creator\graphics\Environment.h"

namespace fs = std::filesystem;

static bool FileExists(const std::string& p){
    std::error_code ec; return fs::exists(p, ec);
}

static std::string s_envJsonPath = "../config/environment.json";
static bool s_inited = false;

// 编辑缓冲
static char  bufSkyboxFolder[512] = {0};
static float skyColor[3]   = {0.05f, 0.08f, 0.12f};
static float ambColor[3]   = {0.15f, 0.15f, 0.15f};
static float sunDir[3]     = {-0.3f, -1.0f, -0.2f};
static float sunColor[3]   = {1.0f, 1.0f, 1.0f};
static float sunIntensity  = 1.0f;
static bool  fogEnabled    = false;
static float fogColor[3]   = {0.6f, 0.7f, 0.8f};
static float fogDensity    = 0.02f;

static void SyncFromEnv(const Environment& env){
    const auto& s = const_cast<Environment&>(env).settings();
    if (!s.skyboxFolder.empty()) {
        std::snprintf(bufSkyboxFolder, sizeof(bufSkyboxFolder), "%s", s.skyboxFolder.c_str());
    } else {
        bufSkyboxFolder[0] = '\0';
    }
    skyColor[0]=s.skyboxColor.r; skyColor[1]=s.skyboxColor.g; skyColor[2]=s.skyboxColor.b;
    ambColor[0]=s.ambientColor.r; ambColor[1]=s.ambientColor.g; ambColor[2]=s.ambientColor.b;
    sunDir[0]=s.sunDirection.x; sunDir[1]=s.sunDirection.y; sunDir[2]=s.sunDirection.z;
    sunColor[0]=s.sunColor.r; sunColor[1]=s.sunColor.g; sunColor[2]=s.sunColor.b;
    sunIntensity = s.sunIntensity;
    fogEnabled   = s.fogEnabled;
    fogColor[0]=s.fogColor.r; fogColor[1]=s.fogColor.g; fogColor[2]=s.fogColor.b;
    fogDensity = s.fogDensity;
}

static void ApplyToEnv(Environment& env){
    auto& s = env.settings();
    s.skyboxFolder = bufSkyboxFolder;
    s.skyboxColor  = { skyColor[0], skyColor[1], skyColor[2] };
    s.ambientColor = { ambColor[0], ambColor[1], ambColor[2] };
    // 归一化太阳方向
    glm::vec3 dir = glm::vec3(sunDir[0], sunDir[1], sunDir[2]);
    if (glm::length(dir) < 1e-4f) dir = glm::vec3(0,-1,0);
    s.sunDirection = glm::normalize(dir);
    s.sunColor     = { sunColor[0], sunColor[1], sunColor[2] };
    s.sunIntensity = sunIntensity;
    s.fogEnabled   = fogEnabled;
    s.fogColor     = { fogColor[0], fogColor[1], fogColor[2] };
    s.fogDensity   = fogDensity;

    env.applySettings();
}

void DrawEnvironmentPanel(Environment& env) {
    if (!s_inited) { s_inited = true; SyncFromEnv(env); }

    if (ImGui::Begin("Environment")) {
        ImGui::TextWrapped("加载/保存 environment.json，选择天空盒或使用纯色背景，并调整光照/雾。");

        // JSON
        ImGui::SeparatorText("配置文件");
        ImGui::TextWrapped("%s", s_envJsonPath.c_str());
        ImGui::SameLine();
        if (ImGui::Button("选择 JSON")) {
            const char* patt[] = {"*.json"};
            const char* sel = tinyfd_openFileDialog("选择 environment.json", s_envJsonPath.c_str(), 1, patt, nullptr, 0);
            if (sel && *sel) s_envJsonPath = sel;
        }
        ImGui::SameLine();
        if (ImGui::Button("加载 JSON")) {
            if (env.loadFromJson(s_envJsonPath)) {
                SyncFromEnv(env);
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("保存 JSON")) {
            ApplyToEnv(env);
            if (!env.saveToJson(s_envJsonPath)) {
                // 失败也不阻断
            }
        }

        // Skybox
        ImGui::SeparatorText("天空盒");
        ImGui::InputText("天空盒文件夹", bufSkyboxFolder, IM_ARRAYSIZE(bufSkyboxFolder));
        ImGui::SameLine();
        if (ImGui::Button("选择文件夹")) {
            const char* sel = tinyfd_selectFolderDialog("选择天空盒文件夹（需包含 right/left/top/bottom/front/back）", bufSkyboxFolder);
            if (sel && *sel) std::snprintf(bufSkyboxFolder, sizeof(bufSkyboxFolder), "%s", sel);
        }
        if (ImGui::Button("清除天空盒（使用纯色）")) {
            bufSkyboxFolder[0] = '\0';
        }
        ImGui::ColorEdit3("纯色背景", skyColor, ImGuiColorEditFlags_Float);

        // Lighting
        ImGui::SeparatorText("光照");
        ImGui::ColorEdit3("环境光", ambColor, ImGuiColorEditFlags_Float);
        ImGui::ColorEdit3("太阳光颜色", sunColor, ImGuiColorEditFlags_Float);
        ImGui::SliderFloat("太阳强度", &sunIntensity, 0.0f, 3.0f, "%.2f");
        ImGui::SliderFloat3("太阳方向", sunDir, -1.0f, 1.0f);

        // Fog
        ImGui::SeparatorText("雾");
        ImGui::Checkbox("启用雾", &fogEnabled);
        ImGui::ColorEdit3("雾颜色", fogColor, ImGuiColorEditFlags_Float);
        ImGui::SliderFloat("雾密度", &fogDensity, 0.0f, 0.2f, "%.3f");

        ImGui::Separator();
        if (ImGui::Button("应用到场景", ImVec2(-1, 0))) {
            ApplyToEnv(env);
        }
    }
    ImGui::End();
}

