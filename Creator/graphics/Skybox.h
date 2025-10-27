#pragma once
#include <string>
#include <glm/glm.hpp>
#include "Shader.h"

// 一个简单的立方体天空盒（立方体贴图）。
// 如果没有提供贴图路径，会用一个纯色立方体作为兜底。
class Skybox {
public:
    Skybox() = default;
    ~Skybox();

    // 从一个文件夹加载 6 张贴图（right/left/top/bottom/front/back，后缀 .jpg/.png 均可）
    // 例如: folder/
    //   right.jpg left.jpg top.jpg bottom.jpg front.jpg back.jpg
    // 成功返回 true；失败时不会崩溃，会退回到纯色天空盒。
    bool loadFromFolder(const std::string& folder);

    // 设置一个纯色天空盒（不依赖外部贴图）
    void setSolidColor(const glm::vec3& rgb);

    // 绘制天空盒。view 会自动去除平移。
    void draw(const glm::mat4& view, const glm::mat4& projection);

    bool isReady() const { return ready; }

private:
    bool createGL();
    void destroyGL();
    bool loadCubemapFiles(const std::string& folder);
    void uploadSolidColor(const glm::vec3& rgb);

private:
    unsigned int vao{0}, vbo{0};
    unsigned int texCube{0};
    Shader skyShader = Shader("../shaders/skybox_vertex.glsl", "../shaders/skybox_fragment.glsl");
    bool ready{false};
};

