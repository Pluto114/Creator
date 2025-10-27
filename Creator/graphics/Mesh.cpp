#include "Mesh.h"
#include <glad/glad.h>
#include <cstddef>
#include <iostream>
#include <limits>
#include <algorithm>

static void PrintGlErrorOnce(const char* where) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "[GL ERROR] at " << where << " : 0x" << std::hex << err << std::dec << "\n";
    }
}

Mesh::Mesh(std::vector<Vertex> v,
           std::vector<unsigned int> i,
           std::vector<Texture> t)
    : vertices(std::move(v)), indices(std::move(i)), textures(std::move(t))
{
    setupMesh();
}

void Mesh::destroyGL() {
    if (EBO) { glDeleteBuffers(1, &EBO); EBO = 0; }
    if (VBO) { glDeleteBuffers(1, &VBO); VBO = 0; }
    if (VAO) { glDeleteVertexArrays(1, &VAO); VAO = 0; }
}

bool Mesh::isGLReady() const {
    if (VAO == 0 || EBO == 0) return false;
    if (!glIsVertexArray(VAO)) return false;
    if (!glIsBuffer(EBO)) return false;
    return true;
}

void Mesh::setupMesh() {
    // 先清理旧对象，允许“重建”
    destroyGL();

    // 允许空网格：只建个 VAO，Draw 会直接返回，不触发错误
    if (vertices.empty() || indices.empty()) {
        std::cerr << "[Mesh] Warning: empty mesh (v=" << vertices.size()
                  << ", i=" << indices.size() << ").\n";
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);
        glBindVertexArray(0);
        VBO = EBO = 0;
        return;
    }

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER,
                 vertices.size() * sizeof(Vertex),
                 vertices.data(),
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 indices.size() * sizeof(unsigned int),
                 indices.data(),
                 GL_STATIC_DRAW);

    // layout
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));

    glBindVertexArray(0);

    if (!isGLReady()) {
        std::cerr << "[Mesh] setupMesh result not GL-ready, VAO=" << VAO
                  << " VBO=" << VBO << " EBO=" << EBO << "\n";
    }
}

void Mesh::Draw(Shader &shader) {
    // 空网格直接返回
    if (indices.empty() || vertices.empty()) return;

    // 纹理绑定（尊重单元上限）
    GLint maxUnits = 0;
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &maxUnits);
    if (maxUnits <= 0) maxUnits = 16;

    unsigned int diffuseNr  = 1;
    unsigned int specularNr = 1;
    unsigned int normalNr   = 1;
    unsigned int heightNr   = 1;

    const size_t bindCount = std::min(textures.size(), static_cast<size_t>(maxUnits));
    for (size_t i = 0; i < bindCount; ++i) {
        glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(i));
        std::string number, name = textures[i].type;

        if      (name == "texture_diffuse")  number = std::to_string(diffuseNr++);
        else if (name == "texture_specular") number = std::to_string(specularNr++);
        else if (name == "texture_normal")   number = std::to_string(normalNr++);
        else if (name == "texture_height")   number = std::to_string(heightNr++);

        shader.setInt((name + number).c_str(), static_cast<int>(i));
        glBindTexture(GL_TEXTURE_2D, textures[i].id);
    }

    // GL 资源失活则自愈一次
    if (!isGLReady()) {
        std::cerr << "[Mesh] VAO/EBO not ready at draw, rebuilding...\n";
        setupMesh();
        if (!isGLReady()) {
            std::cerr << "[Mesh] Rebuild failed, skip draw.\n";
            glActiveTexture(GL_TEXTURE0);
            return;
        }
    }

    // 分批绘制（3 的倍数对齐）
    constexpr GLsizei kMaxIndexPerDraw = 300000;
    const size_t total = indices.size();

    glBindVertexArray(VAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

    size_t start = 0;
    if (start % 3 != 0) start += (3 - (start % 3));

    while (start < total) {
        size_t remain = total - start;
        GLsizei count = static_cast<GLsizei>(std::min<size_t>(kMaxIndexPerDraw, remain));
        count -= (count % 3);
        if (count <= 0) break;

        const void* offset = reinterpret_cast<const void*>(start * sizeof(unsigned int));
        glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, offset);
        PrintGlErrorOnce("glDrawElements(chunk)");

        start += static_cast<size_t>(count);
    }

    glBindVertexArray(0);
    glActiveTexture(GL_TEXTURE0);
}
