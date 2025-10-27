// ===================== Menu.cpp (clean UI, no Mesh/Vertex/Texture) =====================
#include "Menu.h"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include "tinyfiledialogs.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
    #include <direct.h>     // _mkdir
    #include <sys/stat.h>   // _stat
    #include <io.h>         // _access
    #include <objbase.h>    // CoInitializeEx / CoUninitialize
    #include <combaseapi.h>
    #include <Windows.h>
#else
    #include <sys/stat.h>   // mkdir, stat
    #include <sys/types.h>
    #include <unistd.h>     // access
#endif

// -------- 由 main.cpp 定义，用于把“打开文件”的结果传回主循环处理 --------
extern std::string modelPathToLoad;

// -------- 内部实现前置 --------
static void CreateNewProjectWizard_Impl();

// 可变默认资产根目录
static std::string g_AssetRoot = "D:/Creator/Creator/assets/";

// ==========================================================
//                     基础工具（跨平台）
// ==========================================================
static std::string NormalizePath(std::string p) {
    for (char &c : p) if (c == '\\') c = '/';
    return p;
}

static bool PathExists(const std::string& p) {
#ifdef _WIN32
    struct _stat info{}; return _stat(p.c_str(), &info) == 0;
#else
    struct stat info{};  return stat(p.c_str(), &info) == 0;
#endif
}

static bool IsDirectory(const std::string& p) {
#ifdef _WIN32
    struct _stat info{}; if (_stat(p.c_str(), &info) != 0) return false;
    return (info.st_mode & _S_IFDIR) != 0;
#else
    struct stat info{};  if (stat(p.c_str(), &info) != 0) return false;
    return S_ISDIR(info.st_mode);
#endif
}

static bool MkdirOne(const std::string& d) {
    if (d.empty()) return false;
#ifdef _WIN32
    int rc = _mkdir(d.c_str());
#else
    int rc = mkdir(d.c_str(), 0755);
#endif
    if (rc == 0) return true;
    if (errno == EEXIST) return IsDirectory(d);
    return IsDirectory(d);
}

static bool CreateDirsRecursive(std::string path) {
    path = NormalizePath(path);
    if (path.empty()) return false;

    while (!path.empty() && path.back() == '/') path.pop_back();
    if (path.empty()) return false;

    size_t rootLen = 0;
#ifdef _WIN32
    if (path.size() >= 2 && path[1] == ':') {
        rootLen = (path.size() >= 3 && path[2] == '/') ? 3 : 2;
    }
#else
    if (!path.empty() && path[0] == '/') rootLen = 1;
#endif

    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i] == '/') {
            if (i <= rootLen) continue;
            std::string sub = path.substr(0, i);
            if (!IsDirectory(sub) && !MkdirOne(sub)) return false;
        }
    }
    if (!IsDirectory(path) && !MkdirOne(path)) return false;
    return true;
}

static std::string Trim(std::string s) {
    auto notSpace = [](unsigned char ch){ return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}


static std::string SanitizeProjectName(std::string name) {
    name = Trim(name);
#ifdef _WIN32
    const std::string invalid = "<>:\"/\\|?*";
#else
    const std::string invalid = "/\0";
#endif
    for (char &c : name) {
        if (invalid.find(c) != std::string::npos) c = '_';
    }
#ifdef _WIN32
    while (!name.empty() && (name.back() == '.' || name.back() == ' ')) name.pop_back();
    auto upper = name; std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    const char* reserved[] = { "CON","PRN","AUX","NUL",
        "COM1","COM2","COM3","COM4","COM5","COM6","COM7","COM8","COM9",
        "LPT1","LPT2","LPT3","LPT4","LPT5","LPT6","LPT7","LPT8","LPT9"
    };
    for (auto r : reserved) if (upper == r) { name += "_"; break; }
#endif
    return name;
}

static std::string PathJoin(const std::string& a, const std::string& b) {
    if (a.empty()) return NormalizePath(b);
    if (b.empty()) return NormalizePath(a);
    std::string na = NormalizePath(a);
    std::string nb = NormalizePath(b);
    if (na.back() != '/') na += '/';
    return na + nb;
}

static std::string MakeNonConflicting(const std::string& want) {
    if (!PathExists(want)) return want;
    for (int i = 1; i < 1000; ++i) {
        std::string trial = want + "_" + std::to_string(i);
        if (!PathExists(trial)) return trial;
    }
    return want;
}

// ==========================================================
//                 “新建项目”延迟执行机制（帧尾执行）
// ==========================================================
static std::atomic<bool> g_NewWizardPending{false};
static bool g_NewProjectBusy = false;

// 菜单点击时调用：仅设置“待执行”标记
void RequestNewProjectWizard() {
    g_NewWizardPending.store(true, std::memory_order_relaxed);
}

// ------ 文件选择（Open） ------
static std::string SelectModelFile() {
    const char * const filterPatterns[6] = {
        "*.obj", "*.glb", "*.gltf", "*.fbx", "*.stl", "*.dae"
    };
    const char * defaultPath = g_AssetRoot.c_str();

    const char * selectedPath = tinyfd_openFileDialog(
        "选择一个模型文件 (Select a Model File)",
        defaultPath,
        6,
        filterPatterns,
        "3D Model Files",
        0
    );

    if (!selectedPath) {
        std::cout << "文件选择被取消或失败。\n";
        return "";
    } else {
        std::cout << "选择了文件: " << selectedPath << "\n";
        return std::string(selectedPath);
    }
}

// 将加载请求传回主循环处理
void RequestLoadModel(const std::string& path) {
    if (!path.empty()) {
        std::cout << "请求加载模型: " << path << "\n";
        modelPathToLoad = path;
    }
}

// ------ 兼容旧版：无参主菜单（从当前上下文拿窗口） ------
void DrawMainMenu() {
    GLFWwindow* window = glfwGetCurrentContext();

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("文件 (File)")) {
            if (ImGui::MenuItem("新建 (New)", "Ctrl+N")) {
                RequestNewProjectWizard();
            }
            if (ImGui::MenuItem("打开 (Open)", "Ctrl+O")) {
                std::string filePath = SelectModelFile();
                if (!filePath.empty()) {
                    RequestLoadModel(filePath);
                }
            }
            if (ImGui::MenuItem("保存 (Save)", "Ctrl+S")) {
                std::cout << "Menu Item: Save clicked\n";
            }
            ImGui::Separator();
            if (ImGui::MenuItem("退出 (Exit)", "Alt+F4")) {
                if (window) glfwSetWindowShouldClose(window, true);
                std::cout << "Menu Item: Exit clicked\n";
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("设置 (Set)")) {
            if (ImGui::MenuItem("API Key 设置...")) {
                std::cout << "Menu Item: API Key Settings clicked\n";
            }
            if (ImGui::MenuItem("模型输出路径...")) {
                std::cout << "Menu Item: Output Path Settings clicked\n";
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("帮助 (Help)")) {
            if (ImGui::MenuItem("关于 (About)")) {
                std::cout << "Menu Item: About clicked\n";
            }
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}

// ------ 新版：带参数主菜单（目前仅用到 window，用于退出） ------
void DrawMainMenu(GLFWwindow* window,
                  Scene& /*scene*/,
                  Camera& /*camera*/,
                  Environment& /*env*/,
                  Terrain& /*terrain*/,
                  ProjectConfig& /*project*/)
{
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("文件 (File)")) {
            if (ImGui::MenuItem("新建 (New)", "Ctrl+N")) {
                RequestNewProjectWizard();
            }
            if (ImGui::MenuItem("打开 (Open)", "Ctrl+O")) {
                std::string filePath = SelectModelFile();
                if (!filePath.empty()) {
                    RequestLoadModel(filePath);
                }
            }
            if (ImGui::MenuItem("保存 (Save)", "Ctrl+S")) {
                std::cout << "Menu Item: Save clicked\n";
            }
            ImGui::Separator();
            if (ImGui::MenuItem("退出 (Exit)", "Alt+F4")) {
                if (window) glfwSetWindowShouldClose(window, true);
                else {
                    GLFWwindow* ctx = glfwGetCurrentContext();
                    if (ctx) glfwSetWindowShouldClose(ctx, true);
                }
                std::cout << "Menu Item: Exit clicked\n";
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("设置 (Set)")) {
            if (ImGui::MenuItem("API Key 设置...")) {
                std::cout << "Menu Item: API Key Settings clicked\n";
            }
            if (ImGui::MenuItem("模型输出路径...")) {
                std::cout << "Menu Item: Output Path Settings clicked\n";
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("帮助 (Help)")) {
            if (ImGui::MenuItem("关于 (About)")) {
                std::cout << "Menu Item: About clicked\n";
            }
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}

// ------ 帧尾执行：在 glfwSwapBuffers 之后调用 ------
void ProcessDeferredUIActions() {
    if (g_NewWizardPending.exchange(false, std::memory_order_acq_rel)) {
        CreateNewProjectWizard_Impl();
    }
}

// ------ 真正执行“新建项目”向导（帧尾调用） ------
static void CreateNewProjectWizard_Impl() {
    if (g_NewProjectBusy) {
        std::cout << "[New] Wizard already running, ignored.\n";
        return;
    }
    g_NewProjectBusy = true;
    struct Guard { bool* p; ~Guard(){ if(p) *p=false; } } _guard{ &g_NewProjectBusy };

#ifdef _WIN32
    bool needUninit = false;
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (hr == S_OK || hr == S_FALSE)       needUninit = true;
    else if (hr == RPC_E_CHANGED_MODE)     needUninit = false;
#endif

    try {
        std::string defPick = PathJoin(g_AssetRoot, "MyProject");

        const char* pickedC = tinyfd_saveFileDialog(
            "新建项目 (选择父目录并输入项目名)",
            defPick.c_str(),
            0, nullptr,
            "Project Root"
        );
        if (!pickedC) {
#ifdef _WIN32
            if (needUninit) CoUninitialize();
#endif
            std::cout << "[New] 用户取消\n";
            return;
        }

        std::string picked = NormalizePath(pickedC);
        std::string parent = picked;
        std::string projName;
        auto pos = picked.find_last_of('/');
        if (pos != std::string::npos) {
            parent   = picked.substr(0, pos);
            projName = picked.substr(pos + 1);
        } else {
            parent   = g_AssetRoot;
            projName = picked;
        }

        projName = SanitizeProjectName(projName);
        if (projName.empty()) {
            tinyfd_messageBox("错误", "项目名称无效。", "ok", "error", 1);
#ifdef _WIN32
            if (needUninit) CoUninitialize();
#endif
            return;
        }
        if (!PathExists(parent) || !IsDirectory(parent)) {
            tinyfd_messageBox("错误", "所选父目录不存在或不是文件夹。", "ok", "error", 1);
#ifdef _WIN32
            if (needUninit) CoUninitialize();
#endif
            return;
        }

        std::string projectRoot = MakeNonConflicting(PathJoin(parent, projName));
        std::string assetsDir   = PathJoin(projectRoot, "assets");

        if (!CreateDirsRecursive(projectRoot)) {
            tinyfd_messageBox("错误", "创建项目目录失败。", "ok", "error", 1);
#ifdef _WIN32
            if (needUninit) CoUninitialize();
#endif
            return;
        }
        if (!CreateDirsRecursive(assetsDir)) {
            tinyfd_messageBox("错误", "创建资产目录失败。", "ok", "error", 1);
#ifdef _WIN32
            if (needUninit) CoUninitialize();
#endif
            return;
        }

        g_AssetRoot = assetsDir;

        std::string okMsg = std::string("已创建项目：\n") + projectRoot +
                            "\n资产目录：\n" + g_AssetRoot;
        tinyfd_messageBox("完成", okMsg.c_str(), "ok", "info", 1);

        std::cout << "[New] Project created at: " << projectRoot << "\n";
        std::cout << "[New] Asset root set to : " << g_AssetRoot << "\n";
    }
    catch (const std::exception& e) {
        tinyfd_messageBox("异常", (std::string("新建项目异常：\n") + e.what()).c_str(), "ok", "error", 1);
    }
    catch (...) {
        tinyfd_messageBox("异常", "新建项目遇到未知异常。", "ok", "error", 1);
    }

#ifdef _WIN32
    if (needUninit) CoUninitialize();
#endif
}
// ===================== End of file =====================
