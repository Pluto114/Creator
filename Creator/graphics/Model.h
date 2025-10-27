#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <string>

#include <assimp/scene.h> // 需要 Assimp 的类型 aiNode, aiScene, aiMaterial, aiTextureType

#include "Mesh.h"    // 包含我们定义的 Mesh 类
#include <limits>              // ★ 新增
#include <glm/glm.hpp>          // ★ 新增（用于 vec3）

// 前向声明 Shader 类
class Shader;

class Model {
public:
    // 模型数据
    std::vector<Mesh> meshes;            // 模型包含的网格列表
    std::string directory;         // 模型文件所在的目录路径
    std::vector<Texture> textures_loaded; // 用于存储所有已加载的纹理，防止重复加载
    // ★★★ 新增：聚合 AABB
    glm::vec3 aabbMin{ std::numeric_limits<float>::max() };
    glm::vec3 aabbMax{ -std::numeric_limits<float>::max() };
    // 新增：让外部知道是否加载成功
    bool isLoaded() const { return m_loaded; }

    // 构造函数，通过文件路径加载模型
    Model(const std::string& path);

    // 渲染模型 (绘制所有网格)
    void Draw(Shader &shader);

    // ★★★ 新增：本地空间 AABB 访问（用于贴地）
    const glm::vec3& getLocalAABBMin() const { return aabbMin; }
    const glm::vec3& getLocalAABBMax() const { return aabbMax; }

    const glm::vec3& LocalAabbMin() const { return aabbMin; }
    const glm::vec3& LocalAabbMax() const { return aabbMax; }


private:
    // 使用 Assimp 从文件加载模型的核心函数
    bool loadModel(const std::string& path);

         // ← 改为 bool
    Mesh processMesh(aiMesh *mesh, const aiScene *scene);

    bool m_loaded = false;
    // 递归处理 Assimp 场景图中的节点
    void processNode(aiNode *node, const aiScene *scene);

    // 处理 Assimp 网格 (aiMesh)，将其转换为我们自己的 Mesh 对象


    // 【修改点 1】: 为此函数声明添加 const aiScene* scene 参数
    // 加载材质中的纹理
    std::vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, std::string typeName, const aiScene *scene);

    // 【修改点 2】: 添加新函数 TextureFromMemory 的声明
    // 从内存加载内嵌纹理
    unsigned int TextureFromMemory(const aiTexture* texture, bool gamma = false);

    // 从文件加载纹理
    unsigned int TextureFromFile(const char *path, const std::string &directory, bool gamma = false);


};

#endif // MODEL_H



