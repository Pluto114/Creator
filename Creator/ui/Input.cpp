// D:/Creator/ui/Input.cpp

#include "Input.h"
#include "Menu.h"

#include <imgui.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <array>
#include <GLFW/glfw3.h>
#include <thread>
#include <mutex>
#include <functional>
#include <windows.h>
#include <atomic>
#include <cctype>   // isspace
#include <cstdlib>  // atof
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// ================== 配置常量（按需修改） ==================
static const char* kPythonExe        = R"(D:\Creator\.venv\Scripts\python.exe)"; // ← 改成你的实际 venv
static const char* kDeepseekScript   = R"(D:\Creator\Creator\scripts\deepseek_chat.py)";
static const char* kTripoScript      = R"(D:\Creator\Creator\scripts\tripo_batch_processor.py)"; // 注意 Creator\Creator
static const char* kSDLDir           = R"(D:\Creator\Creator\SDL)";
static const char* kCalcBBoxScript   = R"(D:\Creator\Creator\scripts\calculate_bbox.py)";       // 尺寸归一脚本
static const char* kResizedOutputDir = R"(D:\Creator\Creator\assets\test\resized_output)";      // 统一尺度后的输出目录

// 资产目录输入缓冲（ImGui 需要 char*）
static char g_assetsDirBuf[1024] = R"(D:\Creator\Creator\ProjectTest)";

// ================== 全局/静态状态 ==================
static char promptBuffer[2048] =
    "a serene japanese garden with a stone lantern and a small pond";
static std::string statusMessage;
static std::atomic<bool> isProcessing{false};

static bool looks_like_abs_path(const std::string& s) {
    if (s.size() >= 3 && std::isalpha((unsigned char)s[0]) && s[1]==':' && (s[2]=='\\' || s[2]=='/')) return true; // C:\...
    if (!s.empty() && (s[0]=='\\' || s[0]=='/')) return true; // \server\share 或 /abs
    return false;
}

std::mutex ui_data_mutex;
float generationProgress = 0.0f;
std::string generationStatusText;
std::vector<std::string> generatedModelPaths;

static std::string latestSDLPath;         // 最近一次成功生成的 SDL 路径
static std::string latestPlacementsPath;  // 最近一次成功生成的 placements 路径
static std::string latestResizedDir;      // 最近一次统一尺度输出目录

// ================== 小工具函数 ==================

// 同步执行：不通过 cmd /c，直接 CreateProcess 捕获 stdout+stderr
static std::string exec_no_shell_capture(const std::string& exe,
                                         const std::vector<std::string>& args,
                                         int* out_exit_code = nullptr) {
#ifdef _WIN32
    auto quote = [](const std::string& s)->std::string {
        std::string t; t.reserve(s.size()+2+8);
        t.push_back('"');
        for (char c: s) { if (c=='"') t += "\\\""; else t.push_back(c); }
        t.push_back('"'); return t;
    };
    std::string cmdline = quote(exe);
    for (auto& a : args) { cmdline.push_back(' '); cmdline += quote(a); }

    SECURITY_ATTRIBUTES sa; ZeroMemory(&sa, sizeof(sa));
    sa.nLength = sizeof(sa); sa.lpSecurityDescriptor = nullptr; sa.bInheritHandle = TRUE;

    HANDLE rd = NULL, wr = NULL;
    if (!CreatePipe(&rd, &wr, &sa, 0) || !SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0)) {
        return "ERROR: CreatePipe failed";
    }

    STARTUPINFOA si; PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si)); ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    si.hStdOutput = wr;
    si.hStdError  = wr;
    si.dwFlags   |= STARTF_USESTDHANDLES;

    BOOL ok = CreateProcessA(
        exe.c_str(),
        (LPSTR)cmdline.c_str(),
        NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL,
        &si, &pi
    );
    CloseHandle(wr);
    if (!ok) { CloseHandle(rd); return "ERROR: CreateProcess failed"; }

    std::string out;
    std::array<char, 4096> buf{}; DWORD n=0;
    for (;;) {
        if (!ReadFile(rd, buf.data(), (DWORD)buf.size(), &n, NULL) || n==0) break;
        out.append(buf.data(), buf.data()+n);
    }
    CloseHandle(rd);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD ec = 0; GetExitCodeProcess(pi.hProcess, &ec);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);

    while (!out.empty() && (out.back()=='\n' || out.back()=='\r')) out.pop_back();

    if (out_exit_code) *out_exit_code = (int)ec;
    if (ec != 0) return std::string("ERROR: child exit code = ") + std::to_string(ec) + " | " + out;
    return out;
#else
    std::string cmd = "\""+exe+"\"";
    for (auto& a: args) cmd += " \"" + a + "\"";
    std::array<char, 4096> buffer{}; std::string result;
    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) return "ERROR: popen failed";
    while (fgets(buffer.data(), (int)buffer.size(), pipe) != nullptr) result += buffer.data();
    int ec = pclose(pipe);
    while (!result.empty() && (result.back()=='\n' || result.back()=='\r')) result.pop_back();
    if (out_exit_code) *out_exit_code = ec;
    if (ec != 0) return std::string("ERROR: child exit code = ") + std::to_string(ec) + " | " + result;
    return result;
#endif
}

static std::string sanitize_cli_arg(std::string s) {
    for (char& c : s) if (c=='\r' || c=='\n' || c=='\t') c = ' ';
    return s;
}
static std::string trim_copy(std::string s) {
    while (!s.empty() && (s.back()=='\r' || s.back()=='\n' || isspace((unsigned char)s.back()))) s.pop_back();
    size_t i=0; while (i<s.size() && isspace((unsigned char)s[i])) ++i;
    return s.substr(i);
}

// 异步执行命令并按行回调（本实现会阻塞直到子进程结束，名称沿用）
static void executeCommandAsync(const std::string& command,
                                const std::function<void(const std::string&)>& on_line_read) {
#ifdef _WIN32
    HANDLE rd=NULL, wr=NULL;
    SECURITY_ATTRIBUTES sa;
    ZeroMemory(&sa, sizeof(sa));
    sa.nLength = sizeof(sa); sa.lpSecurityDescriptor = nullptr; sa.bInheritHandle = TRUE;

    if (!CreatePipe(&rd, &wr, &sa, 0) ||
        !SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0)) {
        if (on_line_read) on_line_read("ERROR: Pipe creation failed.");
        return;
    }

    PROCESS_INFORMATION pi{}; STARTUPINFOA si{};
    si.cb = sizeof(STARTUPINFO);
    si.hStdError  = wr;
    si.hStdOutput = wr;
    si.dwFlags   |= STARTF_USESTDHANDLES;

    if (!CreateProcessA(NULL, (LPSTR)command.c_str(), NULL, NULL, TRUE,
                        CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        if (on_line_read) on_line_read("ERROR: CreateProcess failed.");
        CloseHandle(wr); CloseHandle(rd); return;
    }
    CloseHandle(wr);

    std::array<char, 256> buffer{}; DWORD n=0; std::string linebuf;
    while (ReadFile(rd, buffer.data(), (DWORD)buffer.size()-1, &n, NULL) && n) {
        buffer[n]='\0'; linebuf += buffer.data();
        size_t pos;
        while ((pos=linebuf.find('\n')) != std::string::npos) {
            std::string line = linebuf.substr(0, pos);
            if (!line.empty() && line.back()=='\r') line.pop_back();
            if (on_line_read && !line.empty()) on_line_read(line);
            linebuf.erase(0, pos+1);
        }
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD ec=0; GetExitCodeProcess(pi.hProcess, &ec);
    if (on_line_read) on_line_read(std::string("STATUS: child exit code = ")+std::to_string(ec));

    CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(rd);
#else
    FILE* pipe = popen((command + " 2>&1").c_str(), "r");
    if (!pipe) { if (on_line_read) on_line_read("ERROR: popen() failed!"); return; }
    std::array<char, 256> buffer{};
    while (fgets(buffer.data(), (int)buffer.size(), pipe) != nullptr) {
        std::string line = buffer.data();
        if (!line.empty() && line.back()=='\n') line.pop_back();
        if (on_line_read && !line.empty()) on_line_read(line);
    }
    pclose(pipe);
#endif
}

// 在指定目录中查找最新的 scene_*.json
static std::string find_latest_sdl_file(const std::string& sdlDir) {
#ifdef _WIN32
    std::string pattern = sdlDir + "\\scene_*.json";
    WIN32_FIND_DATAA ffd{}; HANDLE h = FindFirstFileA(pattern.c_str(), &ffd);
    if (h == INVALID_HANDLE_VALUE) return "";
    FILETIME latest{0,0}; std::string latestPath;
    do {
        if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            FILETIME ft = ffd.ftLastWriteTime;
            if (CompareFileTime(&latest, &ft) < 0) { latest = ft; latestPath = sdlDir + "\\" + ffd.cFileName; }
        }
    } while (FindNextFileA(h, &ffd));
    FindClose(h); return latestPath;
#else
    (void)sdlDir; return "";
#endif
}

// 在指定目录中查找最新的 placements_*.json
static std::string find_latest_placements_file(const std::string& dir) {
#ifdef _WIN32
    std::string pattern = dir + "\\placements_*.json";
    WIN32_FIND_DATAA ffd{}; HANDLE h = FindFirstFileA(pattern.c_str(), &ffd);
    if (h == INVALID_HANDLE_VALUE) return "";
    FILETIME latest{0,0}; std::string latestPath;
    do {
        if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            FILETIME ft = ffd.ftLastWriteTime;
            if (CompareFileTime(&latest, &ft) < 0) { latest = ft; latestPath = dir + "\\" + ffd.cFileName; }
        }
    } while (FindNextFileA(h, &ffd));
    FindClose(h); return latestPath;
#else
    (void)dir; return "";
#endif
}

// 运行 python（我们自己拼 argv；返回 stdout 与退出码）
static bool run_python_capture(const std::string& script,
                               const std::vector<std::string>& args,
                               std::string* out_text,
                               int* out_ec)
{
    std::vector<std::string> argv; argv.reserve(args.size()+1);
    argv.emplace_back(script);
    for (auto& a : args) argv.emplace_back(a);
    int ec = 0;
    std::string out = exec_no_shell_capture(kPythonExe, argv, &ec);
    if (out_text) *out_text = out;
    if (out_ec)   *out_ec   = ec;
    return ec == 0 && (out.rfind("ERROR:", 0) != 0);
}

// 枚举某目录下的 .glb，填充到 UI 列表（便于你看到输出）
static void collect_glb_paths(const std::string& dir, std::vector<std::string>* out)
{
    if (!out) return;
    out->clear();
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return;
    for (auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        auto p = entry.path();
        auto ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".glb") out->push_back(p.string());
    }
}

// ================== UI ==================
void process_input_ui(GLFWwindow* /*window*/) {
    ImGui::Begin("AI Scene Generator##AI_SCENE_PANEL");

    // ---- 每帧只允许绘制一次（防重复调用）----
    static int last_frame = -1;
    static int call_count_this_frame = 0;
    int frame = ImGui::GetFrameCount();
    if (frame != last_frame) { last_frame = frame; call_count_this_frame = 0; }
    call_count_this_frame++;
    if (call_count_this_frame > 1) { ImGui::End(); return; }

    ImGui::PushID("AI_SCENE_PANEL");

    ImGui::Text("输入 Prompt (用于生成场景描述) 或留空以处理上次生成的 SDL：");
    ImGui::InputTextMultiline("##PromptInput", promptBuffer, IM_ARRAYSIZE(promptBuffer),
                              ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 5),
                              ImGuiInputTextFlags_AllowTabInput);

    const bool should_disable_ui = isProcessing.load(std::memory_order_relaxed);
    if (should_disable_ui) ImGui::BeginDisabled(true);

    // ---- 按钮 1：生成 SDL（同步执行）----
    if (ImGui::Button("1. 生成场景描述文件 (SDL)##btn_gen_sdl")) {
        std::string prompt = promptBuffer;
        if (!prompt.empty()) {
            isProcessing.store(true, std::memory_order_relaxed);
            statusMessage = "准备调用 DeepSeek 生成 SDL...";
            generationProgress = 0.0f;

            std::string cleanPrompt = sanitize_cli_arg(prompt);
            int ec = 0;
            std::vector<std::string> argv = { std::string(kDeepseekScript), cleanPrompt };
            std::string out = exec_no_shell_capture(kPythonExe, argv, &ec);

            std::cout << "[SDL] EXE: " << kPythonExe << "\n"
                      << "[SDL] ARG0: " << kDeepseekScript << "\n"
                      << "[SDL] ARG1: " << cleanPrompt << "\n"
                      << "[SDL] OUT: "  << out << std::endl;

            if (out.rfind("ERROR:", 0) == 0 || ec != 0) {
                statusMessage = "DeepSeek 脚本执行失败: " + out;
            } else if (!out.empty() && looks_like_abs_path(out)) {
                statusMessage = "SDL 文件生成成功! 文件路径: " + out;
                latestSDLPath = out;
            } else {
                statusMessage = "DeepSeek 脚本输出异常: " + out;
            }
            isProcessing.store(false, std::memory_order_relaxed);
        } else {
            statusMessage = "请输入 Prompt!";
        }
    }

    // 资产目录输入
    ImGui::InputText(u8"资产目录（自动识别模型）", g_assetsDirBuf, IM_ARRAYSIZE(g_assetsDirBuf));

    // ---- 按钮 2：批量生成（Tripo -> 尺寸归一 -> 自动摆放）----
    if (ImGui::Button("2. 从最新 SDL 文件批量生成模型##btn_batch_tripo")) {
        std::string sdl_file_path = latestSDLPath.empty() ? find_latest_sdl_file(kSDLDir) : latestSDLPath;

        if (!sdl_file_path.empty()) {
            isProcessing.store(true, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(ui_data_mutex);
                generationProgress = 0.0f;
                generationStatusText = "准备启动批量生成...";
                generatedModelPaths.clear();
                latestPlacementsPath.clear();
            }

            std::thread([sdl_file_path]() {
                const std::string command =
                    std::string("\"") + kPythonExe + "\" -u \"" + kTripoScript + "\" \"" + sdl_file_path + "\"";

                auto line_handler = [](const std::string& raw) {
                    std::string line = trim_copy(raw);
                    std::lock_guard<std::mutex> lock(ui_data_mutex);

                    if (line.rfind("PROGRESS", 0) == 0) {
                        auto pos = line.find(':');
                        if (pos != std::string::npos) {
                            std::string v = trim_copy(line.substr(pos+1)); // "x/total"
                            auto sep = v.find('/');
                            if (sep != std::string::npos) {
                                float cur = std::max(0.0f, (float)atof(v.substr(0, sep).c_str()));
                                float tot = std::max(1.0f, (float)atof(v.substr(sep+1).c_str()));
                                generationProgress = (tot > 0.0f) ? (cur / tot) : 0.0f;
                            }
                        }
                    } else if (line.rfind("STATUS", 0) == 0) {
                        auto pos = line.find(':');
                        if (pos != std::string::npos) generationStatusText = trim_copy(line.substr(pos+1));
                    } else if (line.rfind("FILE", 0) == 0) {
                        auto pos = line.find(':');
                        if (pos != std::string::npos) generatedModelPaths.push_back(trim_copy(line.substr(pos+1)));
                    } else if (line.rfind("PLACEMENTS", 0) == 0) { // 新增：解析场景文件路径
                        auto pos = line.find(':');
                        if (pos != std::string::npos) latestPlacementsPath = trim_copy(line.substr(pos+1));
                    } else if (line.rfind("ERROR", 0) == 0) {
                        generationStatusText = line;
                    }
                };

                // 流式读取 tripo 输出（直到子进程结束）
                executeCommandAsync(command, line_handler);

                // —— tripo 完成后，开始统一尺度 —— //
                {
                    std::lock_guard<std::mutex> lock(ui_data_mutex);
                    generationStatusText = "tripo 批量下载完成，开始统一尺度调整（calculate_bbox.py）...";
                }

                // 确保输出目录存在
                std::error_code _ec;
                fs::create_directories(kResizedOutputDir, _ec);

                // 运行 calculate_bbox.py  <scene_json> <input_dir> <output_dir>
                std::string resize_out; int resize_ec = -1;
                bool resize_ok = run_python_capture(
                    kCalcBBoxScript,
                    { sdl_file_path, std::string(g_assetsDirBuf), std::string(kResizedOutputDir) },
                    &resize_out, &resize_ec
                );

                {
                    std::lock_guard<std::mutex> lock(ui_data_mutex);
                    generationStatusText = std::string("尺寸归一脚本退出码: ") + std::to_string(resize_ec);
                    if (!resize_out.empty()) {
                        generationStatusText += " | 日志: " + resize_out.substr(0, 2048);
                    }
                }

                // === 统一尺度完成后的处理 ===
                if (resize_ok) {
                    std::vector<std::string> resizedFiles;
                    collect_glb_paths(kResizedOutputDir, &resizedFiles);

                    const std::string outDir = kResizedOutputDir;
                    const std::string registryPath = outDir + "\\registry_autogen.json";

                    // 确定 placements：优先 Tripo 输出，其次扫描 SDL 目录兜底
                    std::string placementsPath;
                    {
                        std::lock_guard<std::mutex> lock(ui_data_mutex);
                        placementsPath = latestPlacementsPath;
                    }
                    if (placementsPath.empty()) placementsPath = find_latest_placements_file(kSDLDir);
                    if (placementsPath.empty()) placementsPath = sdl_file_path; // 最后兜底：用 SDL（位置多为 0）

                    {
                        std::lock_guard<std::mutex> lock(ui_data_mutex);
                        latestResizedDir = outDir;

                        generatedModelPaths.emplace_back("【统一尺度输出目录】" + outDir);
                        if (!resizedFiles.empty()) {
                            for (auto& f : resizedFiles) generatedModelPaths.emplace_back(f);
                        } else {
                            generatedModelPaths.emplace_back("(提示) 输出目录为空，可能是匹配阈值过高或导出失败。");
                        }

                        if (fs::exists(registryPath)) {
                            generatedModelPaths.emplace_back("【自动生成的 registry】" + registryPath);
                            generationStatusText =
                                "尺寸归一完成（共 " + std::to_string(resizedFiles.size()) +
                                " 个）。准备自动摆放...\n"
                                "placements = " + placementsPath + "\n"
                                "registry   = " + registryPath + "\n"
                                "assetsRoot = " + outDir;
                        } else {
                            generationStatusText =
                                "尺寸归一完成（共 " + std::to_string(resizedFiles.size()) +
                                " 个）。未发现 registry_autogen.json（仍将尝试摆放）...\n"
                                "placements = " + placementsPath + "\n"
                                "assetsRoot = " + outDir;
                        }
                    }

                    // ★ 使用 placements 进行自动摆放（真正执行在 main.cpp 的主线程里）
                    TryAutoPlaceScene(placementsPath, registryPath);

                } else {
                    std::lock_guard<std::mutex> lock(ui_data_mutex);
                    generationStatusText = "尺寸归一失败，请查看日志（上方）";
                }

                // —— 统一收尾 —— //
                {
                    std::lock_guard<std::mutex> lock(ui_data_mutex);
                    if (generationProgress < 1.0f) generationProgress = 1.0f;
                    if (generationStatusText.find("错误") == std::string::npos &&
                        generationStatusText.find("error") == std::string::npos) {
                        generationStatusText += " (任务完成)";
                    }
                }
                isProcessing.store(false, std::memory_order_relaxed);
            }).detach();
        } else {
            statusMessage = "错误: 找不到有效的 SDL 文件路径。请先成功生成一个场景描述文件。";
        }
    }

    if (should_disable_ui) ImGui::EndDisabled();
    if (isProcessing.load(std::memory_order_relaxed)) {
        ImGui::SameLine(); ImGui::TextUnformatted("处理中...");
    }

    // ---- 进度与结果显示 ----
    if (generationProgress > 0.0f) {
        ImGui::Separator();
        ImGui::TextUnformatted("批量生成进度:");
        ImGui::PushID("GenProgress");
        ImGui::ProgressBar(generationProgress, ImVec2(-FLT_MIN, 0));
        ImGui::PopID();

        std::lock_guard<std::mutex> lock(ui_data_mutex);
        if (!generationStatusText.empty())
            ImGui::TextWrapped("状态: %s", generationStatusText.c_str());

        if (!generatedModelPaths.empty()) {
            ImGui::TextUnformatted("已成功生成模型:");
            ImGui::Indent();
            for (const auto& path : generatedModelPaths) {
                ImGui::BulletText("%s", path.c_str());
            }
            ImGui::Unindent();
        }
    } else if (!statusMessage.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("状态: %s", statusMessage.c_str());
    }

    ImGui::PopID();      // AI_SCENE_PANEL
    ImGui::End();        // AI Scene Generator
}
