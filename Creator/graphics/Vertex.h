// D:/Creator/graphics/Vertex.h

#ifndef VERTEX_H
#define VERTEX_H

#include <glm/glm.hpp>// 包含 GLM 头文件


struct Vertex {
    glm::vec3 Position;  // 位置坐标 (x, y, z)
    glm::vec3 Normal;    // 法线向量 (x, y, z) - 用于光照计算
    glm::vec2 TexCoords; // 纹理坐标 (u, v) - 用于贴图采样
    // --- 以下是可选的，但对高级渲染（如法线贴图）很有用 ---
    // glm::vec3 Tangent;   // 切线向量
    // glm::vec3 Bitangent; // 副切线（或称位切线）
};

#endif // VERTEX_H
