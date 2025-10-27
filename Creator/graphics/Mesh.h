#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "Shader.h"

struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

struct Texture {
    unsigned int id = 0;
    std::string type;
    std::string path;
};

class Mesh {
public:
    // CPU 侧数据（必要时可用来重建 GL 资源）
    std::vector<Vertex>        vertices;
    std::vector<unsigned int>  indices;
    std::vector<Texture>       textures;

    // GL 资源
    unsigned int VAO = 0, VBO = 0, EBO = 0;

public:
    Mesh(std::vector<Vertex> vertices,
         std::vector<unsigned int> indices,
         std::vector<Texture> textures);

    // —— 移动语义：在头文件里 inline 定义，避免 LNK2005 ——
    Mesh(Mesh&& rhs) noexcept {
        vertices = std::move(rhs.vertices);
        indices  = std::move(rhs.indices);
        textures = std::move(rhs.textures);
        VAO = rhs.VAO; VBO = rhs.VBO; EBO = rhs.EBO;
        rhs.VAO = rhs.VBO = rhs.EBO = 0;
    }
    Mesh& operator=(Mesh&& rhs) noexcept {
        if (this != &rhs) {
            destroyGL();
            vertices = std::move(rhs.vertices);
            indices  = std::move(rhs.indices);
            textures = std::move(rhs.textures);
            VAO = rhs.VAO; VBO = rhs.VBO; EBO = rhs.EBO;
            rhs.VAO = rhs.VBO = rhs.EBO = 0;
        }
        return *this;
    }

    // 禁止拷贝
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    // —— 析构 inline，避免多重定义 ——
    ~Mesh() { destroyGL(); }

    void Draw(Shader &shader);

private:
    void setupMesh();     // （可重入）重建 VAO/VBO/EBO
    void destroyGL();     // 释放 VAO/VBO/EBO
    bool isGLReady() const; // VAO/EBO 是否有效
};
