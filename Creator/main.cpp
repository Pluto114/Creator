// D:/Creator/main.cpp (最终完美版 - 含自动摆放队列 + 视频翻转修复)

#ifdef _WIN32
#include <windows.h>
#endif

// === ImGuizmo & GLM 扩展（分解矩阵用） ===
// --- ImGui 头（必须在 ImGuizmo 之前） ---
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// --- ImGuizmo 依赖 ImGui 类型，必须放在后面 ---
#include <ImGuizmo.h>

// --- GLM 扩展（只要在包含 matrix_decompose.hpp 之前 define 宏即可） ---
#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "core/Raycast.h"


#include "core/Raycast.h"             // 你之前给的 BuildCenterRay / RayIntersectAABB


#include <thread>
#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <deque>
#include <mutex>

#include "graphics/Environment.h"
#include "graphics/TerrainSnap.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>

#include "ui/Input.h"
#include "ui/Menu.h"
#include "graphics/Shader.h"
#include "graphics/Model.h"
#include "core/Camera.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "graphics/GridRenderer.h"
#include "graphics/Scene.h"
#include "graphics/PlacementsLoader.h"
#include "core/ProjectConfig.h"
#include "ui/SceneBuilder.h"
#include "ui/EnvironmentPanel.h"
#include "graphics/Terrain.h"
#include "ui/TerrainToolsPanel.h"
#include "core/Raycast.h"
#include <ImGuizmo.h>


#ifdef _WIN32
#include <windows.h>
#endif

// 1) 纯 C 小跳板：不创建任何带析构的本地对象，只用 POD 和指针
#ifdef _WIN32


static bool SafeLoadPlacements_SEH_trampoline(const std::string* placementsPath,
                                              Scene* scene,
                                              const PlacementLoadOptions* opts)
{
    BOOL ok = FALSE;
    __try {
        ok = LoadPlacementsIntoScene(*placementsPath, *scene, *opts) ? TRUE : FALSE;
    }
    __except(EXCEPTION_EXECUTE_HANDLER) {
        // 不使用 iostream，避免隐式临时对象
        OutputDebugStringA("[SafeLoad] SEH caught while loading placements.\n");
        ok = FALSE;
    }
    return ok == TRUE;
}
#else
static bool SafeLoadPlacements_SEH_trampoline(const std::string* placementsPath,
                                              Scene* scene,
                                              const PlacementLoadOptions* opts)
{
    // 非 Windows 平台正常 C++ 异常防护（如需）
    try {
        return LoadPlacementsIntoScene(*placementsPath, *scene, *opts);
    } catch (...) {
        return false;
    }
}
#endif

// 2) 外层正常 C++：在这里安全地构造 opts，随后调用小跳板
static bool SafeLoadPlacements_SEH(const std::string& placementsPath,
                                   Scene& scene,
                                   const ProjectConfig& cfg)
{
    PlacementLoadOptions opts;
    opts.assetsRoot           = cfg.assetsRoot;
    opts.registryPath         = cfg.registryPath;
    opts.placeholderModel     = cfg.placeholderModel;
    opts.clearSceneBeforeLoad = true;

    return SafeLoadPlacements_SEH_trampoline(&placementsPath, &scene, &opts);
}

#ifndef GL_DEBUG_OUTPUT
#define GL_DEBUG_OUTPUT 0x92E0
#endif
#ifndef GL_DEBUG_OUTPUT_SYNCHRONOUS
#define GL_DEBUG_OUTPUT_SYNCHRONOUS 0x8242
#endif
#ifndef GL_DEBUG_SEVERITY_NOTIFICATION
#define GL_DEBUG_SEVERITY_NOTIFICATION 0x826B
#endif

// 与平台/加载器无关的函数指针类型
typedef void (APIENTRY *GLDEBUGPROC_T)(GLenum, GLenum, GLuint, GLenum, GLsizei, const GLchar*, const void*);
typedef void (APIENTRY *PFN_GLDEBUGMESSAGECALLBACK)(GLDEBUGPROC_T, const void*);
typedef void (APIENTRY *PFN_GLDEBUGMESSAGECONTROL)(GLenum, GLenum, GLenum, GLsizei, const GLuint*, GLboolean);

// 统一的回调
// ✅ 关键修复：用 length 安全打印，绝不使用 "%s"
static void APIENTRY GlDebugCallbackShim(GLenum source, GLenum type, GLuint id,
                                         GLenum severity, GLsizei length,
                                         const GLchar* message, const void* userParam)
{
    (void)source; (void)type; (void)id; (void)severity; (void)userParam;
    if (message && length > 0) {
        fprintf(stderr, "[GL DEBUG] %.*s\n", (int)length, message);
    } else if (message) {
        // 极少数驱动会传 length=0 但 message 以 \0 结尾，这里做个兜底
        fprintf(stderr, "[GL DEBUG] %s\n", message);
    } else {
        fprintf(stderr, "[GL DEBUG] (null)\n");
    }
}

// === Gizmo 状态 ===
static bool                 g_GizmoEnabled = true;                         // G 开/关
static ImGuizmo::OPERATION  g_GizmoOp      = ImGuizmo::TRANSLATE;          // W/E/R
static ImGuizmo::MODE       g_GizmoMode    = ImGuizmo::WORLD;              // L 切换 WORLD/LOCAL

// 简单的矩阵→TRS 分解，交给 GLM 做（避免改 Scene 接口）
static bool DecomposeTRS_Main(const glm::mat4& M,
                              glm::vec3& outT, glm::vec3& outEulerDeg, glm::vec3& outS)
{
    using namespace glm;
    vec3 skew; vec4 perspective; quat q;
    if (!decompose(M, outS, q, outT, skew, perspective)) return false;
    vec3 eul = degrees(eulerAngles(q));   // 返回 (pitch/x, yaw/y, roll/z)；和你 Scene 里保持一致的 XYZ 顺序
    outEulerDeg = vec3(eul.x, eul.y, eul.z);
    return true;
}


// 与 GLAD/KHR_debug 无关的启用函数：运行时查地址，查不到就跳过
static void EnableGlDebugOutput()
{
    // 先查 core 4.3/KHR 的入口
    PFN_GLDEBUGMESSAGECALLBACK  pDebugMessageCallback  =
        (PFN_GLDEBUGMESSAGECALLBACK)glfwGetProcAddress("glDebugMessageCallback");
    PFN_GLDEBUGMESSAGECONTROL   pDebugMessageControl   =
        (PFN_GLDEBUGMESSAGECONTROL)glfwGetProcAddress("glDebugMessageControl");

    // 如果 core 入口没有，再试 ARB 后缀
    if (!pDebugMessageCallback) {
        pDebugMessageCallback  =
            (PFN_GLDEBUGMESSAGECALLBACK)glfwGetProcAddress("glDebugMessageCallbackARB");
    }
    if (!pDebugMessageControl) {
        pDebugMessageControl   =
            (PFN_GLDEBUGMESSAGECONTROL)glfwGetProcAddress("glDebugMessageControlARB");
    }

    if (pDebugMessageCallback) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        pDebugMessageCallback(GlDebugCallbackShim, nullptr);

        if (pDebugMessageControl) {
            // 屏蔽通知级别噪声（可按需注释掉）
            pDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE,
                                 GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
        }
        fprintf(stderr, "[GL] Debug output enabled (via %s).\n",
                (glfwGetProcAddress("glDebugMessageCallback") ? "core 4.3/KHR" : "ARB"));
    } else {
        fprintf(stderr, "[GL] Debug output NOT available on this loader/context.\n");
    }
}


// --- 全局/静态变量 ---

int screenWidth = 1280;
int screenHeight = 720;
bool cameraMouseControlEnabled = true;
Camera camera(glm::vec3(0.0f, 1.0f, 5.0f));
float lastX = screenWidth / 2.0f;
float lastY = screenHeight / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f;
float lastFrame = 0.0f;
std::string modelPathToLoad = "";
std::string statusMessage = "就绪";
std::atomic<bool> g_resourcesLoaded(false);

// —— Gizmo 编辑期状态（为避免抖动）——
static bool      g_GizmoEditing    = false;     // 当前是否处在拖拽中（上升沿/下降沿检测）
static int       g_GizmoEditingId  = -1;        // 正在编辑的实例 id
static glm::mat4 g_GizmoWorkingM   = glm::mat4(1.0f); // 交互时的工作矩阵

// 保存当前项目配置（供自动摆放缺省使用）
static ProjectConfig g_ProjectCfg;

// -------------------- 自动摆放任务队列（主线程执行） --------------------
struct AutoPlacementTask {
    std::string placementsPath;  // placements_*.json
    std::string registryPath;    // resized_output/registry_autogen.json
    std::string assetsRoot;      // 可留空：留空时用 g_ProjectCfg.assetsRoot
};
static std::mutex g_apMutex;
static std::deque<AutoPlacementTask> g_apQueue;

// 对外 API：UI 线程/后台线程都可以调用 —— 它只负责入队
bool TryAutoPlaceScene(const std::string& placementsJsonPath,
                       const std::string& registryPath)
{
    AutoPlacementTask t;
    t.placementsPath = placementsJsonPath;
    t.registryPath   = registryPath;
    t.assetsRoot     = ""; // 留空交给主线程用项目配置兜底
    {
        std::lock_guard<std::mutex> lock(g_apMutex);
        g_apQueue.push_back(std::move(t));
    }
    std::cout << "[AutoPlace] 已入队：\n"
              << "  placements = " << placementsJsonPath << "\n"
              << "  registry   = " << registryPath       << std::endl;
    return true;
}

// 主线程泵：每帧调用一次；有任务就执行加载
static void PumpAutoPlacement(Scene& scene)
{
    AutoPlacementTask task;
    {
        std::lock_guard<std::mutex> lock(g_apMutex);
        if (g_apQueue.empty()) return;
        task = std::move(g_apQueue.front());
        g_apQueue.pop_front();
    }

    // 组装加载选项
    PlacementLoadOptions opts;
    opts.assetsRoot       = !task.assetsRoot.empty() ? task.assetsRoot : g_ProjectCfg.assetsRoot;
    opts.registryPath     = task.registryPath;               // 指向统一尺寸后的 registry
    opts.placeholderModel = g_ProjectCfg.placeholderModel;
    opts.clearSceneBeforeLoad = true;

    std::cout << "[AutoPlace] 开始加载场景：\n"
              << "  placements = " << task.placementsPath << "\n"
              << "  assetsRoot  = " << opts.assetsRoot     << "\n"
              << "  registry    = " << opts.registryPath   << std::endl;

    if (!LoadPlacementsIntoScene(task.placementsPath, scene, opts)) {
        std::cerr << "[AutoPlace] 失败：无法加载 " << task.placementsPath << std::endl;
    } else {
        std::cout << "[AutoPlace] 成功加载 placements，并实例化到 Scene。\n";
    }
}

// -------------------- 安全投影：多重重载 --------------------

// ① 以 aspect 为输入
static inline glm::mat4 SafePerspective_aspect(float fovY_deg,
                                               float aspect,
                                               float zNear,
                                               float zFar)
{
    if (!std::isfinite(aspect) || aspect <= std::numeric_limits<float>::epsilon()) {
        aspect = 1.0f;
    }
    zNear = std::max(zNear, 0.001f);
    if (!(zFar > zNear)) zFar = zNear + 1.0f;

    return glm::perspective(glm::radians(fovY_deg), aspect, zNear, zFar);
}

// ② 以 width/height 为输入
static inline glm::mat4 SafePerspective_wh(float fovY_deg,
                                           float width, float height,
                                           float zNear,
                                           float zFar)
{
    const float w = std::max(1.0f, width);
    const float h = std::max(1.0f, height);
    const float aspect = w / h;
    return SafePerspective_aspect(fovY_deg, aspect, zNear, zFar);
}

// ③ 便捷 3 参版本（无相机参数时使用：默认 near/far）
static inline glm::mat4 SafePerspective(float fovY_deg,
                                        float width, float height)
{
    constexpr float kNear = 0.1f;
    constexpr float kFar  = 1000.0f;
    return SafePerspective_wh(fovY_deg, width, height, kNear, kFar);
}

// ④ 完整 5 参版本
static inline glm::mat4 SafePerspective(float fovY_deg,
                                        float width, float height,
                                        float zNear, float zFar)
{
    return SafePerspective_wh(fovY_deg, width, height, zNear, zFar);
}

void loadMainResources() {
    std::cout << "[加载线程] 后台资源加载开始..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(6));
    std::cout << "[加载线程] 所有后台资源加载完成！" << std::endl;
    g_resourcesLoaded.store(true);
}

// ★ 启动画面（视频翻转修复）
void playSplashScreenAndLoad(GLFWwindow* window) {
    std::cout << "[主线程] 开始播放启动动画并启动后台加载..." << std::endl;
    std::thread loadingThread(loadMainResources);

    const std::string relativeVideoPath = "../assets/splash.mp4";
    std::filesystem::path absoluteVideoPath = std::filesystem::absolute(relativeVideoPath);
    std::cerr << "[诊断] 视频绝对路径: " << absoluteVideoPath.string() << "\n";
    std::cerr << "[诊断] 文件是否存在? " << (std::filesystem::exists(absoluteVideoPath) ? "是" : "否") << "\n";

    if (!std::filesystem::exists(absoluteVideoPath)) {
        std::cerr << "[错误] 视频文件未找到！请检查路径！\n";
        if (loadingThread.joinable()) loadingThread.join();
        return;
    }

    cv::VideoCapture cap;
    if (!cap.open(absoluteVideoPath.string(), cv::CAP_FFMPEG)) {
        std::cerr << "[错误] 使用 FFMPEG 后端打开视频失败！\n";
        if (loadingThread.joinable()) loadingThread.join();
        return;
    }

    const int VIDEO_WIDTH  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int VIDEO_HEIGHT = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double VIDEO_FPS = cap.get(cv::CAP_PROP_FPS);
    const double FRAME_DURATION = (VIDEO_FPS > 0) ? (1.0 / VIDEO_FPS) : (1.0 / 24.0);
    std::cerr << "[诊断] 视频信息: " << VIDEO_WIDTH << "x" << VIDEO_HEIGHT << " @ " << VIDEO_FPS << " FPS\n";

    Shader splashShader("../shaders/splash_vertex.glsl", "../shaders/splash_fragment.glsl");
    float quadVertices[] = {
        -1.0f,  1.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,

        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f
    };
    unsigned int quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    GLuint videoTexture;
    glGenTextures(1, &videoTexture);
    glBindTexture(GL_TEXTURE_2D, videoTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 VIDEO_WIDTH  > 0 ? VIDEO_WIDTH  : 1280,
                 VIDEO_HEIGHT > 0 ? VIDEO_HEIGHT : 720,
                 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    cv::Mat frame;
    bool animationFinished = false;
    double lastFrameTime = 0.0;
    bool firstFrameLogged = false;

    glDisable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();

        if (!animationFinished && (currentTime - lastFrameTime >= FRAME_DURATION)) {
            lastFrameTime = currentTime;

            if (cap.read(frame) && !frame.empty()) {
                if (!firstFrameLogged) {
                    std::cerr << "[诊断] 成功读取第一帧! 尺寸: " << frame.cols << "x" << frame.rows << ", 通道数: " << frame.channels() << "\n";
                    firstFrameLogged = true;
                }

                // 垂直翻转，修复显示方向
                cv::flip(frame, frame, 0);

                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glBindTexture(GL_TEXTURE_2D, videoTexture);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                frame.cols, frame.rows,
                                GL_BGR, GL_UNSIGNED_BYTE, frame.data);
            } else {
                if (!firstFrameLogged) {
                    std::cerr << "[诊断] cap.read() 首次即失败或返回空帧! 可能是视频编码问题。\n";
                }
                animationFinished = true;
                std::cout << "[主线程] 视频流结束或读取帧失败。" << std::endl;
            }
        }

        if (animationFinished && g_resourcesLoaded.load()) break;

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        splashShader.use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, videoTexture);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    std::cout << "[主线程] 准备进入主程序, 等待加载线程汇合..." << std::endl;
    if (loadingThread.joinable()) loadingThread.join();
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteTextures(1, &videoTexture);
    cap.release();

    glEnable(GL_DEPTH_TEST);
    std::cout << "[主线程] 启动流程结束！" << std::endl;
}

// 回调与输入
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    screenWidth = width;
    screenHeight = height;
}
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    if (!cameraMouseControlEnabled) return;
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos; lastY = ypos;
    camera.ProcessMouseMovement(xoffset, yoffset);
}
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
static inline bool IsUiOrGizmoCapturing() {
    ImGuiIO& io = ImGui::GetIO();
    // ImGui 需要鼠标/键盘，或者 ImGuizmo 正在 hover / 操作，视作占用
    return io.WantCaptureMouse || io.WantCaptureKeyboard || ImGuizmo::IsUsing() || ImGuizmo::IsOver();
}

void processInput(GLFWwindow *window) {
    // ESC 始终可用（即使 UI/Gizmo 正在占用）
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // 如果 UI 或 Gizmo 正在占用输入，则暂停相机 WASD/滚轮/鼠标拖拽
    if (IsUiOrGizmoCapturing())
        return;

    if (cameraMouseControlEnabled) {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.ProcessKeyboard(FORWARD,  deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.ProcessKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.ProcessKeyboard(LEFT,     deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.ProcessKeyboard(RIGHT,    deltaTime);
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)      camera.ProcessKeyboard(UP,   deltaTime);
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) camera.ProcessKeyboard(DOWN, deltaTime);
    }

    // 原有：O 键切换“相机/鼠标”模式（保留）
    static bool oKeyPressedLastFrame = false;
    bool oKeyPressedThisFrame = (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS);
    if (oKeyPressedThisFrame && !oKeyPressedLastFrame) {
        cameraMouseControlEnabled = !cameraMouseControlEnabled;
        if (cameraMouseControlEnabled) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            firstMouse = true;
            std::cout << "鼠标模式: 摄像机控制 (光标已捕获并隐藏)\n";
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            std::cout << "鼠标模式: UI 交互 (光标已显示)\n";
        }
    }
    oKeyPressedLastFrame = oKeyPressedThisFrame;
}


// 放在 glad 初始化成功后、任何 OpenGL 资源创建前：
static void APIENTRY GlDebugCallback(GLenum source, GLenum type, GLuint id,
                                     GLenum severity, GLsizei length,
                                     const GLchar* message, const void* userParam)
{
    (void)source; (void)type; (void)id; (void)severity; (void)length; (void)userParam;
    std::cerr << "[GL DEBUG] " << message << "\n";
}


int main() {
    bool skipPreload = false;
    if (const char* v = std::getenv("CREATOR_SKIP_PLACEMENTS")) {
#ifdef _WIN32
        skipPreload = (std::string(v) == "1" || _stricmp(v, "true") == 0);
#else
        skipPreload = (std::string(v) == "1" || strcasecmp(v, "true") == 0);
#endif
    }

    if (!skipPreload) {
        // 调用上面的 SafeLoadPlacements_SEH(...)
    } else {
        std::cerr << "[Boot] Skipping placements preload (CREATOR_SKIP_PLACEMENTS=1)\n";
    }

#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    if (!glfwInit()) { std::cerr << "错误：GLFW 初始化失败！\n"; return -1; }
    // === Debug Context 可选 ===
    bool wantDebugCtx = false;
    if (const char* v = std::getenv("CREATOR_GL_DEBUG")) {
#ifdef _WIN32
        wantDebugCtx = (std::string(v) == "1" || _stricmp(v, "true") == 0);
#else
        wantDebugCtx = (std::string(v) == "1" || strcasecmp(v, "true") == 0);
#endif
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, wantDebugCtx ? GL_TRUE : GL_FALSE);

    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "AI 3D Creator", NULL, NULL);
    if (!window) { std::cerr << "错误：无法创建 GLFW 窗口！\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // ...glfw/窗口创建之后
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "错误：GLAD 初始化失败！\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // 只装一次安全回调（内部已 length 安全打印）
    EnableGlDebugOutput();

    // ✅ 这里把原先的 if (wantDebugCtx) { glDebugMessageCallback(...<< msg ...) } 整段删掉，
    //    最多保留一个“检查并打印是否处于 debug context”的只读代码：
    if (wantDebugCtx) {
        int flags = 0;
        glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
        if (flags & 0x00000002) {
            std::cerr << "[GL] Debug context is active.\n";
        }
    }


    std::cout << "CWD: " << std::filesystem::current_path().string() << std::endl;
    std::cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "OpenGL Version: "  << glGetString(GL_VERSION)  << std::endl;

    // 启动视频可通过环境变量跳过
    bool skipSplash = false;
    if (const char* v = std::getenv("CREATOR_NO_SPLASH")) {
#ifdef _WIN32
        skipSplash = (std::string(v) == "1" || _stricmp(v, "true") == 0);
#else
        skipSplash = (std::string(v) == "1" || strcasecmp(v, "true") == 0);
#endif
    }
    if (!skipSplash) playSplashScreenAndLoad(window);

    // 初始鼠标模式
    if (cameraMouseControlEnabled) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        std::cout << "初始鼠标模式: 摄像机控制 (光标已捕获并隐藏)\n";
    } else {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        std::cout << "初始鼠标模式: UI 交互 (光标已显示)\n";
    }

    // ==== ImGui ====
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    {
        const char* fontPath = "D:/Creator/Creator/siyuan/SourceHanSerifSC-Regular.otf";
        float fontSize = 18.0f;
        if (!io.Fonts->AddFontFromFileTTF(fontPath, fontSize, nullptr, io.Fonts->GetGlyphRangesChineseFull())) {
            std::cerr << "警告：无法加载中文字体文件: " << fontPath << "，使用默认字体。\n";
            io.Fonts->AddFontDefault();
        }
        io.Fonts->Build();
    }
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    // ==== 资源 ====
    std::cout << "[主线程] 开始创建主程序 OpenGL 资源..." << std::endl;
    Shader modelShader("../shaders/model_vertex.glsl", "../shaders/model_fragment.glsl");
    GridRenderer gridRenderer("../shaders/grid_vertex.glsl", "../shaders/grid_fragment.glsl");
    gridRenderer.backgroundColor = glm::vec3(0.1f, 0.1f, 0.15f);

    Scene scene;
    Environment env;  env.loadFromJson("../config/environment.json");
    Terrain terrain;  terrain.loadFromJson("../config/terrain.json");

    // 项目配置 + 预加载 placements
    {
        const std::string cfgPath = "../config/project.json";
        LoadProjectConfig(cfgPath, g_ProjectCfg);

        if (!skipPreload && !g_ProjectCfg.placementsPath.empty()) {
            if (!SafeLoadPlacements_SEH(g_ProjectCfg.placementsPath, scene, g_ProjectCfg)) {
                std::cerr << "SafeLoad failed, continue without preloaded placements.\n";
            } else {
                std::cout << "Loaded placements from config (safe): "
                          << g_ProjectCfg.placementsPath << std::endl;
            }
        } else if (skipPreload) {
            std::cerr << "[Boot] Skipping placements preload (CREATOR_SKIP_PLACEMENTS=1)\n";
        } else {
            std::cout << "No placementsPath set in project.json. You can still add models via UI.\n";
        }

        PlacementLoadOptions opts;
        opts.assetsRoot          = g_ProjectCfg.assetsRoot;
        opts.registryPath        = g_ProjectCfg.registryPath;
        opts.placeholderModel    = g_ProjectCfg.placeholderModel;
        opts.clearSceneBeforeLoad = true;

        if (!g_ProjectCfg.placementsPath.empty()) {
            if (!LoadPlacementsIntoScene(g_ProjectCfg.placementsPath, scene, opts)) {
                std::cerr << "Failed to load placements from config: " << g_ProjectCfg.placementsPath << std::endl;
            } else {
                std::cout << "Loaded placements from config: " << g_ProjectCfg.placementsPath << std::endl;
            }
        } else {
            std::cout << "No placementsPath set in project.json. You can still add models via UI.\n";
        }
    }
    std::cout << "[主线程] 主程序资源创建完毕！" << std::endl;

    // ====================== 新增：拾取 & Gizmo 防崩全家桶 ======================
    static int   g_SelectedId       = -1;   // 当前命中的实例下标
    static bool  g_EnablePicking    = false; // 开关：拾取
    static bool  g_EnableGizmo      = false; // 开关：Gizmo
    static int   g_GizmoOperation   = 0;    // 0:TRANSLATE, 1:ROTATE, 2:SCALE
    static int g_GizmoMode = 0;    // 0:WORLD, 1:LOCAL

    auto Clamp01 = [](float v){ return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };

    // 计算屏幕中心射线
    auto ComputeCenterRay = [&](const glm::mat4& proj, const glm::mat4& view) -> std::pair<glm::vec3, glm::vec3> {
        glm::mat4 invVP = glm::inverse(proj * view);
        // NDC 中心 (0,0)，用 z=0(near) 与 z=1(far) 两点反投影
        glm::vec4 p0 = invVP * glm::vec4(0.f, 0.f, -1.f, 1.f); // 注意 GLM 的 NDC: z=-1 near, +1 far
        glm::vec4 p1 = invVP * glm::vec4(0.f, 0.f,  1.f, 1.f);
        p0 /= p0.w; p1 /= p1.w;
        glm::vec3 origin = glm::vec3(p0);
        glm::vec3 dir    = glm::normalize(glm::vec3(p1 - p0));
        return {origin, dir};
    };

    // 把实例局部 AABB（model.aabbMin/Max）变换到世界，再取包围 AABB
    auto ComputeWorldAABB = [](const glm::mat4& M, const glm::vec3& localMin, const glm::vec3& localMax,
                               glm::vec3& outMin, glm::vec3& outMax) -> bool {
        // 构造 8 角点
        glm::vec3 mn = localMin, mx = localMax;
        if (!(std::isfinite(mn.x) && std::isfinite(mn.y) && std::isfinite(mn.z))) return false;
        if (!(std::isfinite(mx.x) && std::isfinite(mx.y) && std::isfinite(mx.z))) return false;

        glm::vec3 corners[8] = {
            {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z},
            {mn.x, mx.y, mn.z}, {mx.x, mx.y, mn.z},
            {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z},
            {mn.x, mx.y, mx.z}, {mx.x, mx.y, mx.z}
        };
        glm::vec3 w0 = glm::vec3(M * glm::vec4(corners[0], 1.f));
        glm::vec3 wmin = w0, wmax = w0;
        for (int i=1;i<8;++i) {
            glm::vec3 w = glm::vec3(M * glm::vec4(corners[i], 1.f));
            wmin = glm::min(wmin, w);
            wmax = glm::max(wmax, w);
        }
        outMin = wmin; outMax = wmax;
        return true;
    };

    // 射线与 AABB（世界）相交（slab），返回最近 t>0
    auto RayAabbHitT = [](const glm::vec3& ro, const glm::vec3& rd,
                          const glm::vec3& bmin, const glm::vec3& bmax) -> float {
        const float EPS = 1e-6f;
        float tmin = -1e30f, tmax = 1e30f;
        for (int i=0;i<3;++i) {
            float o = ro[i], d = rd[i], mn = bmin[i], mx = bmax[i];
            if (std::fabs(d) < EPS) {
                if (o < mn || o > mx) return -1.f;  // 平行且在盒外
            } else {
                float t1 = (mn - o) / d;
                float t2 = (mx - o) / d;
                if (t1 > t2) std::swap(t1, t2);
                tmin = std::max(tmin, t1);
                tmax = std::min(tmax, t2);
                if (tmin > tmax) return -1.f;
            }
        }
        return (tmax > 0.f) ? ((tmin > 0.f) ? tmin : tmax) : -1.f;
    };

    // ====================== 主循环 ======================
    while (!glfwWindowShouldClose(window)) {
    // ----------------- 时间步进 -----------------
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    processInput(window);
    glfwPollEvents();

    // ----------------- ImGui 帧起 -----------------
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();

    // ----------------- 顶部/侧边 UI -----------------
    DrawMainMenu();
    DrawSceneBuilderUI(scene);
    DrawEnvironmentPanel(env);

    static bool sAutoSnap = true;
    static SnapRules sSnapRules;
    static bool sSnapInit = false;
    if (!sSnapInit) {
        sSnapInit = true;
        sSnapRules.yEpsilon = 0.02f;
        sSnapRules.onlyRaise = false;
        sSnapRules.ignoreWater = true;
        sSnapRules.categoryYOffset["building"] = 0.02f;
        sSnapRules.categoryYOffset["prop"]     = 0.01f;
        sSnapRules.categoryYOffset["flora"]    = -0.02f;
        sSnapRules.categoryYOffset["fauna"]    = 0.00f;
        sSnapRules.categoryYOffset["terrain"]  = 0.00f;
        sSnapRules.categoryYOffset["water"]    = 0.00f;
    }
    DrawTerrainToolsPanel(sAutoSnap, sSnapRules);
    process_input_ui(window);

    // ----------------- 处理“打开模型”请求 -----------------
    if (!modelPathToLoad.empty()) {
        try {
            statusMessage = "正在加载并添加到场景: " + modelPathToLoad;
            std::cout << statusMessage << std::endl;
            int id = scene.addModelAutoPlace(modelPathToLoad);
            if (id >= 0) std::cout << "模型已添加到场景 (实例ID=" << id << ")\n";
            else         std::cerr << "添加模型失败: " << modelPathToLoad << "\n";
        } catch (const std::exception& e) {
            std::cerr << "添加模型失败: " << modelPathToLoad << " | " << e.what() << "\n";
        }
        modelPathToLoad.clear();
    }

    // 自动摆放任务队列
    PumpAutoPlacement(scene);

    // ----------------- 帧缓冲/清屏 -----------------
    int fbw = 0, fbh = 0;
    glfwGetFramebufferSize(window, &fbw, &fbh);
    if (fbw <= 0 || fbh <= 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        continue;
    }

    glViewport(0, 0, fbw, fbh);
    glClearColor(gridRenderer.backgroundColor.x, gridRenderer.backgroundColor.y, gridRenderer.backgroundColor.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // ----------------- 相机矩阵 -----------------
    const glm::mat4 projection = SafePerspective(camera.Zoom, (float)fbw, (float)fbh);
    const glm::mat4 view       = camera.GetViewMatrix();

    // 天空盒
    env.drawSkybox(view, projection);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    // 地形
    terrain.draw(view, projection, &env);

    // ★ 只有不在使用 Gizmo 时才自动贴地，避免互相影响
    if (sAutoSnap && !ImGuizmo::IsUsing()) {
        SnapSceneToTerrain(scene, terrain, sSnapRules);
    }

    // ----------------- 常规模型绘制 -----------------
    modelShader.use();
    modelShader.setMat4("projection", projection);
    modelShader.setMat4("view", view);
    modelShader.setVec3("viewPos", camera.Position);
    {
        const auto& E = env.settings();
        glm::vec3 sunDir = glm::length(E.sunDirection) > 1e-4f
                         ? glm::normalize(E.sunDirection)
                         : glm::vec3(-0.3f, -1.0f, -0.2f);
        glm::vec3 fakeDirectionalPos = camera.Position - sunDir * 100.0f;
        modelShader.setVec3("lightPos",   fakeDirectionalPos);
        modelShader.setVec3("lightColor", E.sunColor * E.sunIntensity);
    }
    env.applyToShader(modelShader);
    scene.draw(modelShader);

    // ----------------- 屏幕中心拾取（悬停+点击选择） -----------------
    {
        Ray r = BuildCenterRay(view, projection);
        int hovered = scene.pick(r.origin, r.dir, nullptr);
        scene.setHovered(hovered);

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::GetIO().WantCaptureMouse) {
            scene.setSelected(hovered);
        }
    }

    // ----------------- Gizmo 快捷键（基于 ImGui 输入的边沿） -----------------
    {
        if (ImGui::IsKeyPressed(ImGuiKey_W)) g_GizmoOp = ImGuizmo::TRANSLATE;
        if (ImGui::IsKeyPressed(ImGuiKey_E)) g_GizmoOp = ImGuizmo::ROTATE;
        if (ImGui::IsKeyPressed(ImGuiKey_R)) g_GizmoOp = ImGuizmo::SCALE;

        if (ImGui::IsKeyPressed(ImGuiKey_L)) {
            g_GizmoMode = (g_GizmoMode == 0) ? 1 : 0;  // 0: WORLD, 1: LOCAL
        }
        if (ImGui::IsKeyPressed(ImGuiKey_G)) {
            g_GizmoEnabled = !g_GizmoEnabled;
        }
    }

    // ----------------- “Picking/Gizmo” 面板（恢复） -----------------
    {
        ImGui::Begin("Picking/Gizmo");
        ImGui::Checkbox("启用拾取", &g_EnablePicking);
        ImGui::SameLine();
        ImGui::Checkbox("启用Gizmo (G)", &g_EnableGizmo);

        ImGui::RadioButton("移动 (W)", &g_GizmoOperation, 0); ImGui::SameLine();
        ImGui::RadioButton("旋转 (E)", &g_GizmoOperation, 1); ImGui::SameLine();
        ImGui::RadioButton("缩放 (R)", &g_GizmoOperation, 2);

        ImGui::RadioButton("世界", &g_GizmoMode, 0); ImGui::SameLine();
        ImGui::RadioButton("本地 (L)", &g_GizmoMode, 1);

        if (ImGui::Button("清除选择")) scene.setSelected(-1);
        ImGui::Text("当前选中: %d", scene.getSelected());
        ImGui::Text("Gizmo 活动: %s", (ImGuizmo::IsUsing() || ImGuizmo::IsOver()) ? "Yes" : "No");
        ImGui::End();
    }

    // ----------------- 画屏幕中心十字准星 -----------------
    {
        ImDrawList* dl = ImGui::GetForegroundDrawList();
        const float cx = fbw * 0.5f, cy = fbh * 0.5f;
        const ImU32 col = IM_COL32(255, 255, 255, 200);
        dl->AddLine(ImVec2(cx - 8, cy), ImVec2(cx + 8, cy), col, 1.5f);
        dl->AddLine(ImVec2(cx, cy - 8), ImVec2(cx, cy + 8), col, 1.5f);
    }

    // ----------------- 选中高亮（线框二次绘制） -----------------
    {
        const int sel = scene.getSelected();
        if (sel >= 0) {
            const ModelInstance* inst = scene.get(sel);
            if (inst && inst->visible && inst->model) {
                glEnable(GL_POLYGON_OFFSET_LINE);
                glPolygonOffset(-1.0f, -1.0f);
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glLineWidth(2.0f);

                modelShader.use();
                modelShader.setMat4("projection", projection);
                modelShader.setMat4("view", view);
                modelShader.setMat4("model", inst->transform);
                inst->model->Draw(modelShader);

                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glDisable(GL_POLYGON_OFFSET_LINE);
            }
        }
    }

    // ----------------- ImGuizmo 操作（稳定：拖拽中不分解） -----------------
    {
        static bool      sGizmoEditing = false;
        static int       sEditingId    = -1;
        static glm::mat4 sWorkingM     = glm::mat4(1.0f);

        const int selId = scene.getSelected();

        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist(ImGui::GetForegroundDrawList());
        ImGuizmo::SetRect(0.0f, 0.0f, (float)fbw, (float)fbh);

        if (g_GizmoEnabled && selId >= 0) {
            // 选中变化或首次进入：抓一次当前矩阵作为工作矩阵
            if (!sGizmoEditing || sEditingId != selId) {
                sWorkingM     = scene.instanceModelMatrix(selId);
                sGizmoEditing = false;
                sEditingId    = selId;
            }

            glm::mat4 v = view;
            glm::mat4 p = projection;

            ImGuizmo::OPERATION op = ImGuizmo::TRANSLATE;
            if (g_GizmoOperation == 1) op = ImGuizmo::ROTATE;
            else if (g_GizmoOperation == 2) op = ImGuizmo::SCALE;

            ImGuizmo::MODE mode = static_cast<ImGuizmo::MODE>((g_GizmoMode == 0) ? ImGuizmo::WORLD : ImGuizmo::LOCAL);

            bool used = ImGuizmo::Manipulate(
                glm::value_ptr(v),
                glm::value_ptr(p),
                op,
                mode,                               // ★ 显式枚举，修C2664
                glm::value_ptr(sWorkingM),
                nullptr,                            // deltaMatrix
                nullptr, nullptr, nullptr           // snap / bounds
            );

            if (used) {
                // 编辑中：只更新矩阵，避免每帧分解引入抖动
                scene.setInstanceMatrix(selId, sWorkingM, /*updateTRS=*/false);
                sGizmoEditing = true;
            } else {
                // 刚结束编辑：稳定分解一次写回
                if (sGizmoEditing) {
                    scene.setInstanceMatrix(selId, sWorkingM, /*updateTRS=*/true);
                    sGizmoEditing = false;
                }
                // 空闲：跟随场景里矩阵（防止外部改动）
                sWorkingM = scene.instanceModelMatrix(selId);
            }
        }
    }

    // ----------------- 网格/GUI 收尾 -----------------
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    {
        glm::mat4 gridPlaneModel = glm::scale(glm::mat4(1.0f), glm::vec3(200.0f));
        gridRenderer.Draw(view, projection, camera.Position, gridPlaneModel);
    }
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    // ImGui 渲染&交换前后缓冲
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);

    // 延迟 UI 行为（例如“新建项目”对话框）
    ProcessDeferredUIActions();
}




    // ==== 退出 ====
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}



