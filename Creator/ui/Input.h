#pragma once
#include <string>
struct GLFWwindow;

// ---- UI 面板 ----
void process_input_ui(GLFWwindow* window);

// ---- 自动摆放：由 main.cpp 提供实现（主线程任务队列） ----
// 这里仅做声明，Input.cpp 中直接调用它即可。
// 返回 true 表示任务已成功入队（实际加载在主线程进行）。
bool TryAutoPlaceScene(const std::string& placementsJsonPath,
                       const std::string& registryPath);
