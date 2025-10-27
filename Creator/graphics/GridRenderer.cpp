#include "GridRenderer.h"
#include <vector>
#include <iostream>

GridRenderer::GridRenderer(const std::string& vertexShaderPath, const std::string& fragmentShaderPath)
    : gridShader(vertexShaderPath.c_str(), fragmentShaderPath.c_str()) {
    setupGridMesh();
    // 根据你的场景背景色调整
    // backgroundColor = glm::vec3(0.1f, 0.1f, 0.15f); // 例如你的主背景色
}

GridRenderer::~GridRenderer() {
    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridVBO);
}

void GridRenderer::setupGridMesh() {
    // 创建一个非常大的四边形 (2个三角形) 来覆盖地面
    // 顶点只需要位置信息，因为颜色和网格线在片段着色器中计算
    // 这个四边形将在 Y=0 平面，顶点着色器会用模型矩阵将其缩放到很大
    float planeVertices[] = {
        // positions
        -1.0f, 0.0f, -1.0f,
         1.0f, 0.0f, -1.0f,
        -1.0f, 0.0f,  1.0f,
         1.0f, 0.0f,  1.0f
    };

    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridVBO);

    glBindVertexArray(gridVAO);

    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);

    // 顶点位置属性 (location = 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    std::cout << "无限网格几何体已设置。" << std::endl;
}

void GridRenderer::Draw(const glm::mat4& view, const glm::mat4& projection, const glm::vec3& cameraPosition, const glm::mat4& gridPlaneModelMatrix) {
    // 在绘制网格前，可能需要保存和修改一些OpenGL状态，例如深度测试
    // GLboolean last_depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);
    // glDisable(GL_DEPTH_TEST); // 通常无限地面不需要深度写入或测试，或者在最后绘制
    // glDepthMask(GL_FALSE); // 禁止写入深度缓冲

    gridShader.use();
    gridShader.setMat4("view", view);
    gridShader.setMat4("projection", projection);
    gridShader.setMat4("model", gridPlaneModelMatrix); // 网格平面自身的模型变换

    gridShader.setVec3("gridColor", gridColor);
    gridShader.setVec3("backgroundColor", backgroundColor); // 传递背景色
    gridShader.setFloat("majorGridSpacing", majorGridSpacing);
    gridShader.setFloat("minorGridSpacing", minorGridSpacing);
    gridShader.setVec3("cameraPosition", cameraPosition);
    gridShader.setFloat("fadeStartDistance", fadeStartDistance);
    gridShader.setFloat("fadeEndDistance", fadeEndDistance);


    glBindVertexArray(gridVAO);
    // 我们用两个三角形画一个四边形，使用 GL_TRIANGLE_STRIP 需要4个顶点
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    // 恢复OpenGL状态
    // if(last_depth_test_enabled) glEnable(GL_DEPTH_TEST);
    // glDepthMask(GL_TRUE);
}
