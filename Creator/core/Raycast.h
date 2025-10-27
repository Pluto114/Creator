#pragma once
#include <glm/glm.hpp>


struct Ray {
    glm::vec3 origin{};
    glm::vec3 dir{}; // normalized
};


/// Build a world-space ray from the screen center using View/Proj (OpenGL NDC z in [-1,1]).
Ray BuildCenterRay(const glm::mat4& view, const glm::mat4& proj);


/// Classic slab intersection test between a ray and an AABB (local space).
/// Returns true if hit; when hit, outT is the nearest intersection t (>=0).
bool RayIntersectAABB(const glm::vec3& ro, const glm::vec3& rd,
const glm::vec3& bmin, const glm::vec3& bmax,
float& outT);