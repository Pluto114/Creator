#include "Raycast.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>


Ray BuildCenterRay (const glm::mat4& view, const glm::mat4& proj) {
    glm::mat4 invVP = glm::inverse(proj * view);
    glm::vec4 nearNdc(0, 0, -1, 1);
    glm::vec4 farNdc (0, 0, 1, 1);


    glm::vec4 nearW = invVP * nearNdc; nearW /= nearW.w;
    glm::vec4 farW = invVP * farNdc; farW /= farW.w;


    Ray r;
    r.origin = glm::vec3(nearW);
    r.dir = glm::normalize(glm::vec3(farW - nearW));
    return r;
}


bool RayIntersectAABB(const glm::vec3& ro, const glm::vec3& rd,
const glm::vec3& bmin, const glm::vec3& bmax,
float& outT)
{
    // slab
    const glm::vec3 eps(1e-8f);
    glm::vec3 inv = 1.0f / glm::max(glm::abs(rd), eps) * glm::sign(rd);
    glm::vec3 t0 = (bmin - ro) * inv;
    glm::vec3 t1 = (bmax - ro) * inv;
    glm::vec3 tmin = glm::min(t0, t1);
    glm::vec3 tmax = glm::max(t0, t1);
    float tNear = std::max(std::max(tmin.x, tmin.y), tmin.z);
    float tFar = std::min(std::min(tmax.x, tmax.y), tmax.z);
    if (tFar < 0.0f || tNear > tFar) return false;
    outT = (tNear >= 0.0f) ? tNear : tFar;
    return outT >= 0.0f;
}