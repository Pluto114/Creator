// D:/Creator/graphics/Texture.h

#ifndef TEXTURE_H
#define TEXTURE_H

#include <string>

struct Texture {
    unsigned int id;      // OpenGL 纹理对象的 ID
    std::string type;     // 纹理类型，例如 "texture_diffuse", "texture_specular", "texture_normal"
    // 这个类型名通常会用于在着色器中查找对应的 sampler uniform
    std::string path;     // 加载纹理时的原始文件路径（可选，主要用于缓存判断是否已加载）
};

#endif // TEXTURE_H
