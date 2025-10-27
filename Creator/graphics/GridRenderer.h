#ifndef GRIDRENDERER_H
#define GRIDRENDERER_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Shader.h" // 包含你的 Shader 类
#include <string>
#include <vector>

class GridRenderer {
public:
    GridRenderer(const std::string& vertexShaderPath, const std::string& fragmentShaderPath);
    ~GridRenderer();

    // 绘制网格
    // view: 视图矩阵
    // projection: 投影矩阵
    // cameraPosition: 用于雾效计算
    // gridPlaneModelMatrix: 网格平面自身的模型矩阵 (用于定位和缩放平面)
    void Draw(const glm::mat4& view, const glm::mat4& projection, const glm::vec3& cameraPosition, const glm::mat4& gridPlaneModelMatrix);

    // 网格参数设置 (可选的公共接口，或通过uniform直接在Draw中设置)
    glm::vec3 gridColor = glm::vec3(0.3f, 0.3f, 0.3f);
    glm::vec3 backgroundColor = glm::vec3(0.05f, 0.05f, 0.07f); // 与场景清屏色协调
    float majorGridSpacing = 10.0f;
    float minorGridSpacing = 1.0f;
    float fadeStartDistance = 50.0f;
    float fadeEndDistance = 150.0f;

private:
    Shader gridShader;
    unsigned int gridVAO, gridVBO;

    void setupGridMesh(); // 设置用于绘制网格的平面几何体
};

#endif //GRIDRENDERER_H
