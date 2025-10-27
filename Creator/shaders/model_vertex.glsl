#version 330 core
layout (location = 0) in vec3 aPos;      // 顶点位置
layout (location = 1) in vec3 aNormal;   // 顶点法线
layout (location = 2) in vec2 aTexCoords; // 顶点纹理坐标
// layout (location = 3) in vec3 aTangent;   // 如果使用法线贴图需要
// layout (location = 4) in vec3 aBitangent; // 如果使用法线贴图需要

out vec3 FragPos;     // 输出到片段着色器的世界空间位置
out vec3 Normal;      // 输出到片段着色器的世界空间法线
out vec2 TexCoords;   // 输出到片段着色器的纹理坐标

uniform mat4 model;       // 模型矩阵 (模型 -> 世界)
uniform mat4 view;        // 视图矩阵 (世界 -> 视图)
uniform mat4 projection;  // 投影矩阵 (视图 -> 裁剪)

void main()
{
    // 计算世界空间位置
    FragPos = vec3(model * vec4(aPos, 1.0));
    // 计算世界空间法线 (使用法线矩阵避免非统一缩放问题)
    Normal = normalize(mat3(transpose(inverse(model))) * aNormal);
    // 直接传递纹理坐标
    TexCoords = aTexCoords;

    // 计算裁剪空间位置
    gl_Position = projection * view * vec4(FragPos, 1.0);
}