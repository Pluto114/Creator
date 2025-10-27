#include "Scene.h"
#include "Model.h"
#include "Shader.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// 开启 GLM 实验特性（matrix_decompose / quaternion）
#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include <iostream>
#include <memory>

// ----------------- 内部工具 -----------------

static inline glm::mat4 RotFromEulerDegXYZ(const glm::vec3& deg) {
    glm::vec3 rad = glm::radians(deg);
    glm::mat4 R(1.0f);
    R = glm::rotate(R, rad.x, glm::vec3(1,0,0));
    R = glm::rotate(R, rad.y, glm::vec3(0,1,0));
    R = glm::rotate(R, rad.z, glm::vec3(0,0,1));
    return R;
}

glm::mat4 Scene::composeTRS(const glm::vec3& t,
                            const glm::vec3& eulerDegXYZ,
                            const glm::vec3& s) {
    glm::mat4 T = glm::translate(glm::mat4(1.0f), t);
    glm::mat4 R = RotFromEulerDegXYZ(eulerDegXYZ);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), s);
    return T * R * S; // 列主序：p' = T * R * S * p
}

// 稳定分解：强制正尺度，角度归一化到 [-180,180]
bool Scene::decomposeTRSStable(const glm::mat4& M,
                               glm::vec3& outT, glm::vec3& outRotDeg, glm::vec3& outS)
{
    using namespace glm;
    vec3 skew; vec4 persp; quat q;
    vec3 t, s;

    if (!glm::decompose(M, s, q, t, skew, persp)) return false;

    if (s.x < 0) { s.x = -s.x; q = q * angleAxis(pi<float>(), vec3(1,0,0)); }
    if (s.y < 0) { s.y = -s.y; q = q * angleAxis(pi<float>(), vec3(0,1,0)); }
    if (s.z < 0) { s.z = -s.z; q = q * angleAxis(pi<float>(), vec3(0,0,1)); }

    q = normalize(q);
    vec3 euler = degrees(eulerAngles(q)); // XYZ

    auto wrap = [](float a)->float {
        a = fmod(a, 360.0f);
        if (a > 180.0f) a -= 360.0f;
        if (a < -180.0f) a += 360.0f;
        return a;
    };

    outT = t;
    outS = s;
    outRotDeg = vec3(wrap(euler.x), wrap(euler.y), wrap(euler.z));
    return true;
}

// 局部空间 Ray vs AABB（slab）
static inline bool RayIntersectAABB(const glm::vec3& ro, const glm::vec3& rd,
                                    const glm::vec3& bmin, const glm::vec3& bmax,
                                    float& outT)
{
    using namespace glm;
    vec3 rcp;
    rcp.x = (std::abs(rd.x) > 1e-8f) ? (1.0f / rd.x) : 1e8f;
    rcp.y = (std::abs(rd.y) > 1e-8f) ? (1.0f / rd.y) : 1e8f;
    rcp.z = (std::abs(rd.z) > 1e-8f) ? (1.0f / rd.z) : 1e8f;

    vec3 t0 = (bmin - ro) * rcp;
    vec3 t1 = (bmax - ro) * rcp;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);

    float tNear = std::max(std::max(tmin.x, tmin.y), tmin.z);
    float tFar  = std::min(std::min(tmax.x, tmax.y), tmax.z);
    if (tFar < 0.0f || tNear > tFar) return false;

    outT = (tNear >= 0.0f) ? tNear : tFar;
    return outT >= 0.0f;
}

// ----------------- 资源缓存 -----------------

std::shared_ptr<Model> Scene::getOrLoadModel(const std::string& path) {
    // 直接以原始路径作为 key（必要时可自己做 NormalizePath）
    auto it = modelCache.find(path);
    if (it != modelCache.end()) {
        if (auto sp = it->second.lock()) {
            // 要求 Model 提供 isLoaded()
            return sp->isLoaded() ? sp : nullptr;
        }
    }
    auto sp = std::make_shared<Model>(path);
    if (!sp->isLoaded()) {
        std::cerr << "[Scene] 加载模型失败: " << path << std::endl;
        return nullptr;
    }
    modelCache[path] = sp;
    return sp;
}

// ----------------- 添加/移除 -----------------

int Scene::addModel(const std::string& path,
                    const glm::vec3& position,
                    const glm::vec3& eulerXYZ_deg,
                    const glm::vec3& scale,
                    const std::string& instanceName)
{
    auto m = getOrLoadModel(path);
    if (!m) return -1;

    ModelInstance inst;
    inst.model       = std::move(m);
    inst.position    = position;
    inst.rotationDeg = eulerXYZ_deg;
    inst.scale       = scale;
    inst.transform   = composeTRS(inst.position, inst.rotationDeg, inst.scale);
    inst.name        = instanceName.empty() ? path : instanceName;
    inst.visible     = true;

    instances.emplace_back(std::move(inst));
    return static_cast<int>(instances.size() - 1);
}

int Scene::addModelAutoPlace(const std::string& path)
{
    const int   col     = 5;
    const float spacing = 2.5f;
    int idx = autoPlaceCount++;
    int r = idx / col;
    int c = idx % col;
    glm::vec3 pos = glm::vec3(c * spacing, 0.0f, r * spacing);
    return addModel(path, pos, glm::vec3(0), glm::vec3(1), "auto");
}

void Scene::remove(int instanceId)
{
    if (instanceId < 0 || instanceId >= static_cast<int>(instances.size())) return;
    instances[instanceId].visible = false;
    instances[instanceId].model.reset();
}

void Scene::clear()
{
    for (auto& i : instances) { i.visible = false; i.model.reset(); }
    instances.clear();
}

// ----------------- 绘制 -----------------

void Scene::draw(Shader& shader) const
{
    for (int i = 0; i < static_cast<int>(instances.size()); ++i) {
        const auto& inst = instances[i];
        if (!inst.visible || !inst.model) continue;

        shader.setMat4("model", inst.transform);

        int hl = 0;
        if (i == hoveredId)  hl = 1;
        if (i == selectedId) hl = 2;
        shader.setInt("uHighlightState", hl);

        inst.model->Draw(shader);
    }
}

// ----------------- 访问实例 -----------------

ModelInstance* Scene::get(int instanceId)
{
    if (instanceId < 0 || instanceId >= static_cast<int>(instances.size())) return nullptr;
    return &instances[instanceId];
}

const ModelInstance* Scene::get(int instanceId) const
{
    if (instanceId < 0 || instanceId >= static_cast<int>(instances.size())) return nullptr;
    return &instances[instanceId];
}

// ----------------- 拾取 -----------------

int Scene::pick(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float* outDist) const
{
    int   bestId = -1;
    float bestT  = 1e30f;

    for (int i = 0; i < static_cast<int>(instances.size()); ++i) {
        const auto& ins = instances[i];
        if (!ins.visible || !ins.model) continue;

        glm::mat4 invM = glm::inverse(ins.transform);
        glm::vec3 roL = glm::vec3(invM * glm::vec4(rayOrigin, 1.0f));
        glm::vec3 rdL = glm::normalize(glm::vec3(invM * glm::vec4(rayDir,    0.0f)));

        float t = -1.0f;

        // 你的 Model 如果没有 LocalAabbMin/Max()，就改成 aabbMin/aabbMax
        const glm::vec3 bmin = ins.model->LocalAabbMin();
        const glm::vec3 bmax = ins.model->LocalAabbMax();

        if (RayIntersectAABB(roL, rdL, bmin, bmax, t)) {
            if (t >= 0.0f && t < bestT) {
                bestT  = t;
                bestId = i;
            }
        }
    }
    if (outDist) *outDist = (bestId >= 0 ? bestT : -1.0f);
    return bestId;
}

// ----------------- TRS/矩阵 接口 -----------------

bool Scene::getInstanceTRS(int id, glm::vec3& pos, glm::vec3& rotDeg, glm::vec3& scl) const
{
    if (id < 0 || id >= static_cast<int>(instances.size())) return false;
    const auto& m = instances[id];
    pos    = m.position;
    rotDeg = m.rotationDeg;
    scl    = m.scale;
    return true;
}

bool Scene::setInstanceTRS(int id, const glm::vec3& pos, const glm::vec3& rotDeg, const glm::vec3& scl)
{
    if (id < 0 || id >= static_cast<int>(instances.size())) return false;
    auto& m = instances[id];
    m.position    = pos;
    m.rotationDeg = rotDeg;
    m.scale       = scl;
    m.transform   = composeTRS(m.position, m.rotationDeg, m.scale);
    return true;
}

glm::mat4 Scene::instanceModelMatrix(int id) const
{
    if (id < 0 || id >= static_cast<int>(instances.size())) return glm::mat4(1.0f);
    return instances[id].transform;
}

bool Scene::setInstanceMatrix(int id, const glm::mat4& M, bool updateTRS)
{
    if (id < 0 || id >= static_cast<int>(instances.size())) return false;
    auto& inst = instances[id];
    inst.transform = M;
    if (updateTRS) {
        glm::vec3 t, rdeg, s;
        if (decomposeTRSStable(M, t, rdeg, s)) {
            inst.position    = t;
            inst.rotationDeg = rdeg;
            inst.scale       = s;
        }
    }
    return true;
}
