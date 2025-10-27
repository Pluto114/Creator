#pragma once
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <glm/glm.hpp>

class Shader;
class Model;

struct ModelInstance {
    std::shared_ptr<Model> model;
    glm::mat4 transform{1.0f};       // 最终用于绘制/拾取的模型矩阵
    glm::vec3 position{0.0f};
    glm::vec3 rotationDeg{0.0f};     // XYZ 欧拉角（度）
    glm::vec3 scale{1.0f};
    std::string name;
    bool visible{true};
};

class Scene {
public:
    Scene() = default;

    int addModel(const std::string& path,
                 const glm::vec3& position,
                 const glm::vec3& eulerXYZ_deg,
                 const glm::vec3& scale,
                 const std::string& instanceName = "");

    int  addModelAutoPlace(const std::string& path);
    void remove(int instanceId);
    void clear();

    void draw(Shader& shader) const;

    ModelInstance*       get(int instanceId);
    const ModelInstance* get(int instanceId) const;
    int  size() const { return static_cast<int>(instances.size()); }

    // 屏幕中心/鼠标射线拾取（ray in world space）
    int  pick(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float* outDist) const;

    // TRS 访问/写入
    bool getInstanceTRS(int id, glm::vec3& pos, glm::vec3& rotDeg, glm::vec3& scl) const;
    bool setInstanceTRS(int id, const glm::vec3& pos, const glm::vec3& rotDeg, const glm::vec3& scl);

    // 直接用矩阵（供 ImGuizmo 编辑时调用）
    glm::mat4 instanceModelMatrix(int id) const;
    bool      setInstanceMatrix(int id, const glm::mat4& M, bool updateTRS);

    // 高亮/选择
    void setHovered(int id)  { hoveredId  = id; }
    void setSelected(int id) { selectedId = id; }
    int  getSelected() const { return selectedId; }
    int  getHovered()  const { return hoveredId; }

private:
    static glm::mat4 composeTRS(const glm::vec3& t, const glm::vec3& eulerDegXYZ, const glm::vec3& s);
    static bool      decomposeTRSStable(const glm::mat4& M,
                                        glm::vec3& outT, glm::vec3& outRotDeg, glm::vec3& outS);

    std::shared_ptr<Model> getOrLoadModel(const std::string& path);

private:
    std::vector<ModelInstance> instances;
    std::unordered_map<std::string, std::weak_ptr<Model>> modelCache;

    int hoveredId  = -1;
    int selectedId = -1;
    int autoPlaceCount = 0;
};
