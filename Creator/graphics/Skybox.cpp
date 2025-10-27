#include "Skybox.h"
#include <glad/glad.h>
#include <vector>
#include <string>
#include <filesystem>
#include <iostream>

// 注意：这里不要再定义 STB_IMAGE_IMPLEMENTATION
// 你的工程已经在别处（如 Model.cpp）实现过一次
#include "stb_image.h"

namespace fs = std::filesystem;

static bool fileExists(const std::string& p) {
    std::error_code ec; return fs::exists(p, ec);
}

Skybox::~Skybox() { destroyGL(); }

bool Skybox::createGL() {
    if (vao) return true;

    float verts[] = {
        -1,  1, -1,  -1, -1, -1,   1, -1, -1,   1, -1, -1,   1,  1, -1,  -1,  1, -1,
        -1, -1,  1,  -1, -1, -1,  -1,  1, -1,  -1,  1, -1,  -1,  1,  1,  -1, -1,  1,
         1, -1, -1,   1, -1,  1,   1,  1,  1,   1,  1,  1,   1,  1, -1,   1, -1, -1,
        -1, -1,  1,  -1,  1,  1,   1,  1,  1,   1,  1,  1,   1, -1,  1,  -1, -1,  1,
        -1,  1, -1,   1,  1, -1,   1,  1,  1,   1,  1,  1,  -1,  1,  1,  -1,  1, -1,
        -1, -1, -1,  -1, -1,  1,   1, -1, -1,   1, -1, -1,  -1, -1,  1,   1, -1,  1
    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glBindVertexArray(0);

    glGenTextures(1, &texCube);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texCube);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    return true;
}

void Skybox::destroyGL() {
    if (texCube) glDeleteTextures(1, &texCube);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    texCube = 0; vbo = 0; vao = 0;
    ready = false;
}

bool Skybox::loadFromFolder(const std::string& folder) {
    if (!createGL()) return false;

    if (!loadCubemapFiles(folder)) {
        std::cerr << "[Skybox] loadCubemapFiles failed, fallback to solid color.\n";
        uploadSolidColor(glm::vec3(0.05f, 0.08f, 0.12f)); // 深蓝兜底
    }

    ready = true;
    return true;
}

void Skybox::setSolidColor(const glm::vec3& rgb) {
    if (!createGL()) return;
    uploadSolidColor(rgb);
    ready = true;
}

bool Skybox::loadCubemapFiles(const std::string& folder) {
    static const char* names[] = { "right", "left", "top", "bottom", "front", "back" };
    static const char* exts[]  = { ".jpg", ".png", ".jpeg", ".bmp", ".tga" };

    std::vector<std::string> files(6);
    for (int i=0;i<6;i++) {
        bool found = false;
        for (auto ext: exts) {
            std::string p = folder + "/" + names[i] + ext;
            if (fileExists(p)) { files[i] = p; found = true; break; }
        }
        if (!found) {
            std::cerr << "[Skybox] missing face: " << names[i] << " in folder " << folder << "\n";
            return false;
        }
    }

    stbi_set_flip_vertically_on_load(false);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texCube);
    for (int i=0;i<6;i++) {
        int w,h,n;
        unsigned char* data = stbi_load(files[i].c_str(), &w,&h,&n, 3);
        if (!data) {
            std::cerr << "[Skybox] stbi_load failed: " << files[i] << "\n";
            glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
            return false;
        }
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        stbi_image_free(data);
    }
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    return true;
}

void Skybox::uploadSolidColor(const glm::vec3& rgb) {
    unsigned char px[3] = {
        (unsigned char)glm::clamp(rgb.r*255.0f, 0.0f, 255.0f),
        (unsigned char)glm::clamp(rgb.g*255.0f, 0.0f, 255.0f),
        (unsigned char)glm::clamp(rgb.b*255.0f, 0.0f, 255.0f)
    };
    glBindTexture(GL_TEXTURE_CUBE_MAP, texCube);
    for (int i=0;i<6;i++) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, px);
    }
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void Skybox::draw(const glm::mat4& view, const glm::mat4& projection) {
    if (!ready) return;

    glm::mat4 viewNoTrans = glm::mat4(glm::mat3(view));

    glDepthMask(GL_FALSE);
    glDepthFunc(GL_LEQUAL);

    skyShader.use();
    skyShader.setMat4("projection", projection);
    skyShader.setMat4("view", viewNoTrans);
    skyShader.setInt("skybox", 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texCube);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
}

