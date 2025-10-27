#include "TerrainSnap.h"
#include "Scene.h"
#include "Terrain.h"
#include "Model.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <limits>

static glm::vec3 TransformPoint(const glm::mat4& M, const glm::vec3& p) {
    glm::vec4 r = M * glm::vec4(p, 1.0f);
    return glm::vec3(r);
}

static float WorldMinYOfAABB(const glm::mat4& M, const glm::vec3& aabbMin, const glm::vec3& aabbMax) {
    // 8 个角
    glm::vec3 c[8] = {
        {aabbMin.x, aabbMin.y, aabbMin.z},
        {aabbMax.x, aabbMin.y, aabbMin.z},
        {aabbMin.x, aabbMax.y, aabbMin.z},
        {aabbMax.x, aabbMax.y, aabbMin.z},
        {aabbMin.x, aabbMin.y, aabbMax.z},
        {aabbMax.x, aabbMin.y, aabbMax.z},
        {aabbMin.x, aabbMax.y, aabbMax.z},
        {aabbMax.x, aabbMax.y, aabbMax.z},
    };
    float minY = std::numeric_limits<float>::max();
    for (int i=0;i<8;i++) {
        glm::vec3 w = TransformPoint(M, c[i]);
        if (w.y < minY) minY = w.y;
    }
    return minY;
}

void SnapSceneToTerrain(Scene& scene, const Terrain& terrain, const SnapRules& rules) {
    const size_t N = scene.size();
    for (size_t i=0; i<N; ++i) {
        auto inst = scene.get((int)i);
        if (!inst || !inst->visible || !inst->model) continue;

        // 取本地 AABB
        const glm::vec3 aabbMin = inst->model->getLocalAABBMin();
        const glm::vec3 aabbMax = inst->model->getLocalAABBMax();

        // 当前世界矩阵 & 世界位置
        glm::mat4 M = inst->transform;
        glm::vec3 worldPos = glm::vec3(M[3]); // 平移分量

        // 该实例当前“脚底”的世界 y
        float minY = WorldMinYOfAABB(M, aabbMin, aabbMax);

        // 地形高度（以实例的世界 xz 为采样位置）
        float hT = terrain.heightAtWorld(worldPos.x, worldPos.z) + rules.yEpsilon;

        // 需要的平移量
        float deltaY = hT - minY;
        if (rules.onlyRaise && deltaY < 0.0f) continue; // 仅抬高：若已经高于地形则不动

        // 应用：只改 Y 平移分量
        M[3].y += deltaY;
        inst->transform = M;
    }
}

