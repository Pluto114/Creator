#pragma once
class Environment;

// 在 ImGui 中绘制“环境设置”面板：加载/保存 JSON、选择天空盒文件夹、编辑参数并应用。
void DrawEnvironmentPanel(Environment& env);

