#pragma once
class Scene;

// 在 ImGui 窗口中画出“场景生成器”面板。
// 点击按钮会：调用 Python 生成 placements.json -> 载入到传入的 Scene。
void DrawSceneBuilderUI(Scene& scene);

