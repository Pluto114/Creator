// --- !!! 添加 stb_image 实现宏定义 (仅在此 .cpp 文件中) !!!
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h> // <--- 包含 stb_image 头文件 (确保路径正确)

#include "Model.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
// 可选：若你想使用 AI_CONFIG_ 开头的配置宏，再启用下面一行
// #include <assimp/config.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <limits>

// 作用域仅限本翻译单元，避免与别处同名
namespace {
    inline std::string NormalizePath(std::string s) {
        for (char &c : s) if (c == '\\') c = '/';
        // 合并重复的 '/'
        size_t pos = 0;
        while ((pos = s.find("//", pos)) != std::string::npos) s.erase(pos, 1);
        return s;
    }
}

// 构造函数实现
Model::Model(const std::string& path) {
    m_loaded = loadModel(path);
}

// 绘制函数
void Model::Draw(Shader &shader) {
    for (unsigned int i = 0; i < meshes.size(); i++) {
        meshes[i].Draw(shader);
    }
}

// --- Assimp 加载核心逻辑 ---
// 返回 true/false，失败不再让外部误用
bool Model::loadModel(const std::string& path) {
    aabbMin = glm::vec3( std::numeric_limits<float>::max() );
    aabbMax = glm::vec3( -std::numeric_limits<float>::max() );

    std::cout << "开始加载模型: " << path << std::endl;
    Assimp::Importer importer;

    // 更稳妥的处理标志（去掉容易触发的切线计算；先把网格读进来再按需生成）
    const unsigned int PP =
      aiProcess_Triangulate
    | aiProcess_JoinIdenticalVertices
    | aiProcess_GenSmoothNormals
    | aiProcess_CalcTangentSpace
    | aiProcess_FlipUVs
    | aiProcess_SortByPType
    | aiProcess_ImproveCacheLocality
    | aiProcess_OptimizeMeshes
    | aiProcess_RemoveRedundantMaterials
    | aiProcess_FindInvalidData           // 查找非法数据
    | aiProcess_ValidateDataStructure;    // 严格验证

    const aiScene* scene = importer.ReadFile(path, PP);



    if(!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
        std::cerr << "错误::ASSIMP::读取模型失败: " << importer.GetErrorString() << std::endl;
        // 关键：返回 false，让上层跳过该模型
        return false;
    }

    directory = path.substr(0, path.find_last_of('/'));
    if (directory == path || directory.empty()) {
        size_t last_slash = path.find_last_of('\\');
        directory = (last_slash != std::string::npos) ? path.substr(0, last_slash) : ".";
    }

    std::cout << "模型目录: " << directory << std::endl;
    std::cout << "开始处理模型节点..." << std::endl;
    processNode(scene->mRootNode, scene);
    std::cout << "模型加载处理完毕，共包含 " << meshes.size() << " 个网格。" << std::endl;

    // 即使成功读到 scene，也防御一下：没有任何网格视为失败
    if (meshes.empty()) {
        std::cerr << "警告：模型无网格，跳过: " << path << std::endl;
        return false;
    }

    // 兜底：如果 AABB 仍然是未初始化值，给一个小盒避免后续求交出 NaN
    if (!(aabbMin.x <= aabbMax.x && aabbMin.y <= aabbMax.y && aabbMin.z <= aabbMax.z)) {
        aabbMin = glm::vec3(-0.5f);
        aabbMax = glm::vec3( 0.5f);
    }

    return true;
}

// 递归处理节点（无需关心节点矩阵，Assimp 已经 PreTransform 过）
void Model::processNode(aiNode *node, const aiScene *scene) {
    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.emplace_back(processMesh(mesh, scene));
    }
    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene);
    }
}

// 处理网格数据
// 处理网格数据（请用此版本完全覆盖你现有的 processMesh）
Mesh Model::processMesh(aiMesh *mesh, const aiScene *scene) {
    std::vector<Vertex>  vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;

    const unsigned int vcount = mesh ? mesh->mNumVertices : 0;
    const unsigned int fcount = mesh ? mesh->mNumFaces    : 0;

    std::cout << "  处理网格: "
              << ((mesh && mesh->mName.length > 0) ? mesh->mName.C_Str() : "unnamed")
              << ", 顶点数: " << vcount
              << ", 面数: "   << fcount << std::endl;

    // ---------- 1) 顶点 ----------
    vertices.reserve(vcount);
    for (unsigned int i = 0; i < vcount; ++i) {
        Vertex v{};

        // 位置
        aiVector3D p = mesh->mVertices[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            // 清洗 NaN/Inf
            p = aiVector3D(0,0,0);
        }
        v.Position = glm::vec3(p.x, p.y, p.z);

        // 法线
        if (mesh->HasNormals()) {
            aiVector3D n = mesh->mNormals[i];
            if (!std::isfinite(n.x) || !std::isfinite(n.y) || !std::isfinite(n.z)) n = aiVector3D(0,1,0);
            v.Normal = glm::normalize(glm::vec3(n.x, n.y, n.z));
        } else {
            v.Normal = glm::vec3(0.0f, 1.0f, 0.0f);
        }

        // UV（仅通道0）
        if (mesh->HasTextureCoords(0)) {
            aiVector3D t = mesh->mTextureCoords[0][i];
            // 不做翻转，这里和 aiProcess_FlipUVs 配合
            v.TexCoords = glm::vec2(t.x, t.y);
        } else {
            v.TexCoords = glm::vec2(0.0f);
        }

        // 更新 AABB
        if (v.Position.x < aabbMin.x) aabbMin.x = v.Position.x;
        if (v.Position.y < aabbMin.y) aabbMin.y = v.Position.y;
        if (v.Position.z < aabbMin.z) aabbMin.z = v.Position.z;
        if (v.Position.x > aabbMax.x) aabbMax.x = v.Position.x;
        if (v.Position.y > aabbMax.y) aabbMax.y = v.Position.y;
        if (v.Position.z > aabbMax.z) aabbMax.z = v.Position.z;

        vertices.push_back(v);
    }

    // ---------- 2) 索引（强校验） ----------
    // 如果某个 face 不是三角形（极少见，万一导入器没三角化），直接跳过
    size_t droppedFaces = 0, badIndexFaces = 0;
    indices.reserve(static_cast<size_t>(fcount) * 3);

    for (unsigned int i = 0; i < fcount; ++i) {
        const aiFace& face = mesh->mFaces[i];
        if (face.mNumIndices != 3) {
            ++droppedFaces;
            continue;
        }
        // 越界检测：任意一个 index >= 顶点数，都丢弃这个三角形
        unsigned int i0 = face.mIndices[0];
        unsigned int i1 = face.mIndices[1];
        unsigned int i2 = face.mIndices[2];
        if (i0 >= vcount || i1 >= vcount || i2 >= vcount) {
            ++badIndexFaces;
            continue;
        }
        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i2);
    }

    if (droppedFaces > 0) {
        std::cerr << "  [WARN] 非三角面被剔除: " << droppedFaces << "\n";
    }
    if (badIndexFaces > 0) {
        std::cerr << "  [WARN] 检测到越界索引的三角面，已全部丢弃: " << badIndexFaces << "\n";
    }
    if (indices.empty()) {
        std::cerr << "  [ERROR] 有效索引为空（可能模型损坏）。该网格将被跳过以避免 GPU 崩溃。\n";
        // 返回一个空网格，setupMesh 会安全处理
        return Mesh(std::vector<Vertex>{}, std::vector<unsigned int>{}, std::vector<Texture>{});
    }

    // ---------- 3) 材质与纹理 ----------
    if (mesh->mMaterialIndex >= 0 && scene) {
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        std::cout << "    处理材质索引: " << mesh->mMaterialIndex << std::endl;

        // 漫反射 / 镜面 / 法线 / glTF 常见通道
        auto diffuseMaps  = loadMaterialTextures(material, aiTextureType_DIFFUSE,  "texture_diffuse",  scene);
        auto specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular", scene);
        auto normalMaps   = loadMaterialTextures(material, aiTextureType_HEIGHT,   "texture_normal",   scene);
        if (normalMaps.empty())
            normalMaps   = loadMaterialTextures(material, aiTextureType_NORMALS,   "texture_normal",   scene);
        auto baseColor   = loadMaterialTextures(material, aiTextureType_BASE_COLOR,         "texture_basecolor", scene);
        auto metallic    = loadMaterialTextures(material, aiTextureType_METALNESS,          "texture_metallic",  scene);
        auto roughness   = loadMaterialTextures(material, aiTextureType_DIFFUSE_ROUGHNESS,  "texture_roughness", scene);
        auto ao          = loadMaterialTextures(material, aiTextureType_AMBIENT_OCCLUSION,  "texture_ao",        scene);

        textures.insert(textures.end(), diffuseMaps.begin(),  diffuseMaps.end());
        textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
        textures.insert(textures.end(), normalMaps.begin(),   normalMaps.end());
        textures.insert(textures.end(), baseColor.begin(),    baseColor.end());
        textures.insert(textures.end(), metallic.begin(),     metallic.end());
        textures.insert(textures.end(), roughness.begin(),    roughness.end());
        textures.insert(textures.end(), ao.begin(),           ao.end());
    } else {
        std::cout << "    提示: 网格无材质或 scene 为空，跳过纹理加载。\n";
    }

    std::cout << "  网格数据和纹理处理完毕 (顶点: " << vertices.size()
              << ", 索引: " << indices.size()
              << ", 纹理数: " << textures.size() << ")\n";

    return Mesh(vertices, indices, textures);
}




// ---------------- 纹理加载（保持你原有实现） ----------------

// 【新增函数】从内存加载内嵌纹理
unsigned int Model::TextureFromMemory(const aiTexture* texture, bool gamma) {
    if (!texture) return 0;

    unsigned int textureID = 0;
    glGenTextures(1, &textureID);

    int width = 0, height = 0, nrComponents = 0;
    unsigned char* data = nullptr;

    if (texture->mHeight == 0) {
        stbi_set_flip_vertically_on_load(false);
        data = stbi_load_from_memory(
            reinterpret_cast<unsigned char*>(texture->pcData),
            static_cast<int>(texture->mWidth),
            &width, &height, &nrComponents, 0
        );
    } else {
        width  = static_cast<int>(texture->mWidth);
        height = static_cast<int>(texture->mHeight);
        nrComponents = 4;
        const size_t byteSize = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
        data = static_cast<unsigned char*>(malloc(byteSize));
        if (data) {
            const unsigned char* src = reinterpret_cast<const unsigned char*>(texture->pcData);
            memcpy(data, src, byteSize);
        }
    }

    if (data) {
        GLenum format = (nrComponents == 4) ? GL_RGBA : (nrComponents == 3 ? GL_RGB : GL_RED);
        GLenum internalFormat = (format == GL_RGBA)
                                ? (gamma ? GL_SRGB_ALPHA : GL_RGBA)
                                : (format == GL_RGB ? (gamma ? GL_SRGB : GL_RGB) : GL_RED);

        glBindTexture(GL_TEXTURE_2D, textureID);

        GLint prevUnpack = 4;
        glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevUnpack);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glPixelStorei(GL_UNPACK_ALIGNMENT, prevUnpack);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        std::cout << "内嵌纹理加载成功 (ID: " << textureID << ", "
                  << width << "x" << height << ", comp=" << nrComponents << ")\n";
    } else {
        std::cerr << "错误: 内嵌纹理加载失败 | stb_image 错误: " << stbi_failure_reason() << std::endl;
        glDeleteTextures(1, &textureID);
        textureID = 0;
    }

    if (texture->mHeight == 0) stbi_image_free(data); else free(data);
    glBindTexture(GL_TEXTURE_2D, 0);
    return textureID;
}

std::vector<Texture> Model::loadMaterialTextures(aiMaterial *mat, aiTextureType type,
                                                 std::string typeName, const aiScene *scene) {
    std::vector<Texture> textures;

    const unsigned int cnt = mat->GetTextureCount(type);
    for (unsigned int i = 0; i < cnt; i++) {
        aiString str;
        if (mat->GetTexture(type, i, &str) != AI_SUCCESS) continue;

        std::string texturePath = NormalizePath(std::string(str.C_Str()));
        if (texturePath.empty()) continue;

        bool skip = false;
        for (unsigned int j = 0; j < textures_loaded.size(); j++) {
            if (NormalizePath(textures_loaded[j].path) == texturePath) {
                textures.push_back(textures_loaded[j]);
                skip = true;
                break;
            }
        }
        if (skip) continue;

        Texture texture;
        bool applyGamma = (typeName == "texture_diffuse" || typeName == "texture_basecolor");

        if (!texturePath.empty() && texturePath[0] == '*') {
            int textureIndex = -1;
            try { textureIndex = std::stoi(texturePath.substr(1)); } catch (...) { textureIndex = -1; }
            if (!scene || textureIndex < 0 || textureIndex >= static_cast<int>(scene->mNumTextures)) {
                std::cerr << "警告: 内嵌纹理索引无效: " << texturePath << std::endl;
                continue;
            }
            const aiTexture* embeddedTexture = scene->mTextures[textureIndex];
            texture.id = TextureFromMemory(embeddedTexture, applyGamma);
        } else {
            texture.id = TextureFromFile(texturePath.c_str(), this->directory, applyGamma);
        }

        if (texture.id != 0) {
            texture.type = typeName;
            texture.path = texturePath;
            textures.push_back(texture);
            textures_loaded.push_back(texture);
        } else {
            std::cerr << "警告: 加载类型 '" << typeName << "' 的纹理失败，路径: " << texturePath << std::endl;
        }
    }
    return textures;
}

unsigned int Model::TextureFromFile(const char *path, const std::string &directory, bool gamma) {
    std::string filename = std::string(path);
    if (filename.find(":") == std::string::npos && filename[0] != '/' && filename[0] != '\\') {
        filename = directory + '/' + filename;
    }
    for (char &c : filename) if (c == '\\') c = '/';
    std::cout << "DEBUG: Texture path resolved: " << filename << std::endl;

    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    stbi_set_flip_vertically_on_load(false);
    unsigned char *data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);

    if (data) {
        GLenum format, internalFormat;
        if      (nrComponents == 1) { format = internalFormat = GL_RED; }
        else if (nrComponents == 3) { format = GL_RGB;  internalFormat = gamma ? GL_SRGB       : GL_RGB; }
        else if (nrComponents == 4) { format = GL_RGBA; internalFormat = gamma ? GL_SRGB_ALPHA : GL_RGBA; }
        else {
            std::cerr << "警告: 不支持的纹理通道数 " << nrComponents << " 在文件: " << filename << std::endl;
            stbi_image_free(data);
            glDeleteTextures(1, &textureID);
            return 0;
        }

        glBindTexture(GL_TEXTURE_2D, textureID);

        GLint prevUnpack = 4;
        glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevUnpack);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glPixelStorei(GL_UNPACK_ALIGNMENT, prevUnpack);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        std::cout << "纹理加载成功: " << filename << " (ID: " << textureID << ")" << std::endl;
    } else {
        std::cerr << "错误: 纹理加载失败，路径: " << filename
                  << " | stb_image 错误: " << stbi_failure_reason() << std::endl;
        textureID = 0;
    }

    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 0);
    return textureID;
}
