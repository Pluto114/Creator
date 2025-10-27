#pragma once
#include <string>

// 前置声明，避免在头文件里引入一堆图形相关头
struct GLFWwindow;
class Scene;
class Camera;
class Environment;
class Terrain;
class ProjectConfig;

// 触发“新建项目”向导（在帧尾执行，避免与 ImGui/GLFW 事件循环冲突）
void RequestNewProjectWizard();

// 请求从路径加载一个模型（把路径传回主循环去真正加载）
void RequestLoadModel(const std::string& path);

// 旧版（兼容）无参主菜单
void DrawMainMenu();

// 新版（可扩展）主菜单
void DrawMainMenu(GLFWwindow* window,
                  Scene& scene,
                  Camera& camera,
                  Environment& env,
                  Terrain& terrain,
                  ProjectConfig& project);

// 在帧尾（SwapBuffers 之后）调用，用来执行需要脱离 ImGui 事件循环的操作
void ProcessDeferredUIActions();
