#version 330 core
layout (location = 0) in vec3 aPos; // 网格平面的顶点位置 (模型空间)

uniform mat4 model;      // 网格平面的模型矩阵 (用于缩放/定位网格平面本身)
uniform mat4 view;       // 视图矩阵
uniform mat4 projection; // 投影矩阵

out vec3 WorldPos_FS;    // 输出到片段着色器的世界坐标

void main() {
    WorldPos_FS = vec3(model * vec4(aPos, 1.0));
    gl_Position = projection * view * vec4(WorldPos_FS, 1.0);
}