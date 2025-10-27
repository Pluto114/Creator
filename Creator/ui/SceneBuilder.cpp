#include "SceneBuilder.h"

#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <filesystem>
#include <algorithm>

#include <imgui.h>
#include "tinyfiledialogs.h"

#include "PlacementsLoader.h"
#include "ProjectConfig.h"

// --------- 小工具 ---------
namespace {
    namespace fs = std::filesystem;

    static std::string Trim(const std::string& s) {
        auto b = s.find_first_not_of(" \t\r\n");
        auto e = s.find_last_not_of(" \t\r\n");
        if (b == std::string::npos) return "";
        return s.substr(b, e - b + 1);
    }
    static std::string Quote(const std::string& s) {
    #ifdef _WIN32
        if (s.find_first_of(" \t\"") != std::string::npos) {
            return "\"" + s + "\"";
        }
        return s;
    #else
        if (s.find(' ') != std::string::npos) return "\"" + s + "\"";
        return s;
    #endif
    }
    static bool FileExists(const std::string& p) {
        std::error_code ec; return fs::exists(p, ec) && fs::is_regular_file(p, ec);
    }
    static bool DirExists(const std::string& p) {
        std::error_code ec; return fs::exists(p, ec) && fs::is_directory(p, ec);
    }
    static std::string JoinPath(const std::string& a, const std::string& b){
    #ifdef _WIN32
        const char sep = '\\';
    #else
        const char sep = '/';
    #endif
        if(a.empty()) return b;
        if(b.empty()) return a;
        if(a.back()==sep) return a + b;
        return a + sep + b;
    }

    static std::string ExecAndCapture(const std::string& cmd, int* exitCode = nullptr) {
    #ifdef _WIN32
        FILE* pipe = _popen(cmd.c_str(), "r");
    #else
        FILE* pipe = popen(cmd.c_str(), "r");
    #endif
        if (!pipe) { if(exitCode) *exitCode = -1; return ""; }
        std::string out; char buf[4096];
        while (fgets(buf, sizeof(buf), pipe)) out += buf;
    #ifdef _WIN32
        int code = _pclose(pipe);
    #else
        int code = pclose(pipe);
    #endif
        if (exitCode) *exitCode = code;
        return out;
    }

    // 取外部脚本输出最后一行非空文本
    static std::string TakeLastNonEmptyLine(const std::string& text) {
        std::istringstream iss(text);
        std::string line, last;
        while (std::getline(iss, line)) {
            line = Trim(line);
            if (!line.empty()) last = line;
        }
        return last;
    }
}

// --------- UI 内部状态（静态变量保存跨帧） ---------
static bool s_initialized = false;

// 注意：ImGui::InputText 需要固定缓冲，这里用 char[]
static char s_pythonExe[1024] = "D:/Creator/.venv/Scripts/python.exe"; // 可在 UI 改
static std::string s_assetsRoot;            // 资产根目录（原始）
static std::string s_assetsRootResized;     // 统一尺度后的资产根目录（默认 = assetsRoot/resized_output）
static std::string s_sdlPath;               // SDL (scene_*.json)
static std::string s_placementsPath;        // placements_*.json
static std::string s_status;                // 状态/日志

// 常量脚本路径（以 cmake-build-* 为工作目录）
static const char* kScript_SDL2Layout   = "../scripts/sdl_to_layout.py";
static const char* kScript_CalcBBox     = "../scripts/calculate_bbox.py";

// 初次调用时，从 project.json 读默认值
static void EnsureInitialized() {
    if (s_initialized) return;
    s_initialized = true;

    ProjectConfig cfg;
    const std::string cfgPath = "../config/project.json";
    LoadProjectConfig(cfgPath, cfg);

    if (!cfg.assetsRoot.empty()) s_assetsRoot = cfg.assetsRoot;
    if (!cfg.placementsPath.empty()) s_placementsPath = cfg.placementsPath;

    // 默认统一尺度输出目录
    if (!s_assetsRoot.empty()) {
        s_assetsRootResized = JoinPath(s_assetsRoot, "resized_output");
    }
#ifdef _WIN32
    // 如果你有不同的 venv，可在 UI 修改 s_pythonExe
#endif
}

// 运行 Python 生成 placements，并返回输出路径（脚本会打印一行绝对路径）
static bool RunPythonGeneratePlacements(const std::string& sdlPath,
                                        std::string& outPlacementsPath,
                                        std::string& outLog)
{
    if (!FileExists(kScript_SDL2Layout)) {
        outLog = std::string("脚本不存在: ") + kScript_SDL2Layout;
        return false;
    }
    if (!FileExists(sdlPath)) {
        outLog = "SDL 文件不存在: " + sdlPath;
        return false;
    }
    std::string cmd = Quote(s_pythonExe) + " " + Quote(kScript_SDL2Layout) + " " + Quote(sdlPath);
    int code = 0;
    std::string output = ExecAndCapture(cmd, &code);
    outLog = "命令: " + cmd + "\n输出:\n" + output;
    if (code != 0) {
        outLog += "\n返回码: " + std::to_string(code);
        return false;
    }
    std::string last = TakeLastNonEmptyLine(output);
    if (last.empty() || !FileExists(last)) {
        outLog += "\n未解析到有效输出路径。";
        return false;
    }
    outPlacementsPath = last;
    return true;
}

// 运行 calculate_bbox.py 统一尺度
static bool RunPythonCalcBBox(const std::string& sdlPath,
                              const std::string& assetsRoot,
                              const std::string& outputDir,
                              std::string& outLog)
{
    if (!FileExists(kScript_CalcBBox)) {
        outLog = std::string("脚本不存在: ") + kScript_CalcBBox;
        return false;
    }
    if (!FileExists(sdlPath)) {
        outLog = "SDL 文件不存在: " + sdlPath;
        return false;
    }
    if (!DirExists(assetsRoot)) {
        outLog = "资产目录不存在: " + assetsRoot;
        return false;
    }
    std::error_code ec;
    fs::create_directories(outputDir, ec);

    // 本版 calculate_bbox.py 按：<json_file> <input_dir> <output_dir>
    std::string cmd = Quote(s_pythonExe) + " " + Quote(kScript_CalcBBox) + " "
                    + Quote(sdlPath) + " " + Quote(assetsRoot) + " " + Quote(outputDir);
    int code = 0;
    std::string output = ExecAndCapture(cmd, &code);
    outLog = "命令: " + cmd + "\n输出:\n" + output + "\n返回码: " + std::to_string(code);
    return (code == 0);
}

void DrawSceneBuilderUI(Scene& scene) {
    EnsureInitialized();

    if (ImGui::Begin("AI Scene Generator")) {
        ImGui::TextWrapped("选择资产目录与 SDL 文件，然后一键生成布局并载入。");

        // ---------------- Python ----------------
        ImGui::SeparatorText("Python");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##python", s_pythonExe, IM_ARRAYSIZE(s_pythonExe))) {
            // no-op; s_pythonExe 直接被修改
        }

        // ---------------- 资产目录 ----------------
        ImGui::SeparatorText("资产目录");
        ImGui::Text("原始资产根：");
        ImGui::TextWrapped("%s", s_assetsRoot.empty() ? "(未选择)" : s_assetsRoot.c_str());
        if (ImGui::Button("选择资产目录")) {
            const char* sel = tinyfd_selectFolderDialog("选择资产根目录 (assetsRoot)", s_assetsRoot.empty()? nullptr : s_assetsRoot.c_str());
            if (sel && *sel) {
                s_assetsRoot = sel;
                if (s_assetsRootResized.empty())
                    s_assetsRootResized = JoinPath(s_assetsRoot, "resized_output");
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("打开原始目录")) {
        #ifdef _WIN32
            if (!s_assetsRoot.empty()) {
                std::string cmd = "explorer " + Quote(s_assetsRoot);
                system(cmd.c_str());
            }
        #endif
        }

        ImGui::Text("统一尺度输出目录：");
        ImGui::TextWrapped("%s", s_assetsRootResized.empty() ? "(未设置)" : s_assetsRootResized.c_str());
        ImGui::SameLine();
        if (ImGui::Button("更改输出目录")) {
            const char* sel = tinyfd_selectFolderDialog("选择统一尺度输出目录 (resized_output)", s_assetsRootResized.empty()? nullptr : s_assetsRootResized.c_str());
            if (sel && *sel) s_assetsRootResized = sel;
        }
        ImGui::SameLine();
        if (ImGui::Button("打开输出目录")) {
        #ifdef _WIN32
            if (!s_assetsRootResized.empty()) {
                std::string cmd = "explorer " + Quote(s_assetsRootResized);
                system(cmd.c_str());
            }
        #endif
        }

        // ---------------- SDL 文件 ----------------
        ImGui::SeparatorText("SDL 文件");
        ImGui::TextWrapped("%s", s_sdlPath.empty() ? "(未选择)" : s_sdlPath.c_str());
        if (ImGui::Button("选择 SDL (.json)")) {
            const char* patterns[] = { "*.json" };
            const char* sel = tinyfd_openFileDialog("选择 SDL JSON", nullptr, 1, patterns, nullptr, 0);
            if (sel && *sel) s_sdlPath = sel;
        }

        // ---------------- 当前 placements ----------------
        ImGui::SeparatorText("场景布局（placements）");
        ImGui::TextWrapped("%s", s_placementsPath.empty() ? "(暂无)" : s_placementsPath.c_str());
        ImGui::SameLine();
        if (ImGui::Button("查看 placements 文件")) {
            if (!s_placementsPath.empty()) {
            #ifdef _WIN32
                std::string cmd = "explorer /select," + Quote(s_placementsPath);
                system(cmd.c_str());
            #endif
            }
        }

        // ---------------- 一键流程按钮 ----------------
        ImGui::Separator();
        bool canRun = !s_sdlPath.empty() && !s_assetsRoot.empty() && !s_assetsRootResized.empty();
        if (!canRun) ImGui::BeginDisabled();
        if (ImGui::Button("一键：生成 placements → 统一尺度 → 载入场景", ImVec2(-1, 0))) {
            std::string logAll;

            // 1) 生成 placements
            {
                std::string newPlacements, log;
                if (RunPythonGeneratePlacements(s_sdlPath, newPlacements, log)) {
                    s_placementsPath = newPlacements;
                    logAll += "[生成 placements 成功]\n" + log + "\n";
                } else {
                    s_status = "[生成 placements 失败]\n" + log;
                    goto _END_UPDATE_STATUS;
                }
            }

            // 2) 统一尺度（calculate_bbox.py）
            {
                std::string log;
                if (RunPythonCalcBBox(s_sdlPath, s_assetsRoot, s_assetsRootResized, log)) {
                    logAll += "[统一尺度 成功]\n" + log + "\n";
                } else {
                    s_status = "[统一尺度 失败]\n" + log;
                    goto _END_UPDATE_STATUS;
                }
            }

            // 3) 载入场景（优先使用统一尺度后的目录与 registry）
            {
                PlacementLoadOptions opts;
                opts.assetsRoot = s_assetsRootResized; // ★ 关键：改为统一尺度后的目录
                const std::string registryPath = JoinPath(s_assetsRootResized, "registry_autogen.json");
                if (FileExists(registryPath)) {
                    opts.registryPath = registryPath;   // 若存在，则优先映射 id->GLB
                    logAll += "[registry] 使用: " + registryPath + "\n";
                } else {
                    logAll += "[registry] 未找到 registry_autogen.json，改用 assetsRoot 兜底匹配。\n";
                }

                // 占位模型（可选）
                const std::string placeholderA = JoinPath(s_assetsRootResized, "placeholder.glb");
                const std::string placeholderB = JoinPath(s_assetsRoot,        "placeholder.glb");
                if (FileExists(placeholderA)) opts.placeholderModel = placeholderA;
                else if (FileExists(placeholderB)) opts.placeholderModel = placeholderB;

                opts.clearSceneBeforeLoad = true;

                if (LoadPlacementsIntoScene(s_placementsPath, scene, opts)) {
                    logAll += "[载入] 成功载入: " + s_placementsPath + "\n";
                } else {
                    logAll += "[载入] 失败，请检查控制台错误。\n";
                }

                // 写回配置，方便下次直接打开
                ProjectConfig cfg;
                cfg.assetsRoot      = s_assetsRootResized;   // 下次直接指向统一尺度目录
                cfg.placeholderModel= opts.placeholderModel;
                cfg.placementsPath  = s_placementsPath;
                SaveProjectConfig("../config/project.json", cfg);
            }

            s_status = logAll;

        _END_UPDATE_STATUS:
            (void)0;
        }
        if (!canRun) ImGui::EndDisabled();

        // ---------------- 状态显示 ----------------
        ImGui::SeparatorText("状态 / 日志");
        ImGui::TextWrapped("%s", s_status.empty()? "(无)" : s_status.c_str());
    }
    ImGui::End();
}
