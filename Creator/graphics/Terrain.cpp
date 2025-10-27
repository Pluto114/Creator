#include "Terrain.h"
#include "Environment.h" // 仅用于把环境 uniform 传给地形 shader
#include "nlohmann/json.hpp"
#include "stb_image.h"   // 注意：不要在本文件 define IMPLEMENTATION

#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <random>

using json = nlohmann::json;
namespace fs = std::filesystem;

// ---------------- 工具函数 ----------------
static bool fileExists(const std::string& p){ std::error_code ec; return fs::exists(p, ec); }
static float lerp(float a, float b, float t){ return a + (b - a) * t; }
static float fade(float t){ return t*t*(3.0f - 2.0f*t); } // smootherstep-ish

// 简单的栅格值噪声（非经典 Perlin），足够预览
static float valueNoise2D(int x, int z, unsigned int seed=1337) {
    // 32-bit hash
    unsigned int h = (unsigned int)(x) * 374761393u + (unsigned int)(z) * 668265263u + seed*982451653u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return (float)(h & 0x00FFFFFF) / (float)0x01000000; // [0,1)
}

static float noise2D(float fx, float fz, float freq, int octaves=4, float persistence=0.5f){
    float amp = 1.0f;
    float sum = 0.0f;
    float norm= 0.0f;
    float x = fx * freq;
    float z = fz * freq;

    for(int o=0;o<octaves;o++){
        int xi = (int)std::floor(x);
        int zi = (int)std::floor(z);
        float tx = x - xi;
        float tz = z - zi;

        float v00 = valueNoise2D(xi,   zi);
        float v10 = valueNoise2D(xi+1, zi);
        float v01 = valueNoise2D(xi,   zi+1);
        float v11 = valueNoise2D(xi+1, zi+1);

        float vx0 = lerp(v00, v10, fade(tx));
        float vx1 = lerp(v01, v11, fade(tx));
        float v = lerp(vx0, vx1, fade(tz)); // [0,1)

        sum  += v * amp;
        norm += amp;
        amp  *= persistence;
        x    *= 2.0f;
        z    *= 2.0f;
    }
    sum /= std::max(norm, 1e-6f);
    return sum*2.0f - 1.0f; // [-1,1]
}

// 由高度图计算法线（中心差分）
static glm::vec3 calcNormal(const std::vector<float>& h, int res, int ix, int iz,
                            float stepX, float stepZ) {
    auto idx = [&](int x, int z){ return z * (res+1) + x; };
    int ix0 = std::max(ix-1,0), ix1 = std::min(ix+1,res);
    int iz0 = std::max(iz-1,0), iz1 = std::min(iz+1,res);
    float hx0 = h[idx(ix0, iz)];
    float hx1 = h[idx(ix1, iz)];
    float hz0 = h[idx(ix, iz0)];
    float hz1 = h[idx(ix, iz1)];
    glm::vec3 dx = glm::vec3(2.0f*stepX, hx1 - hx0, 0.0f);
    glm::vec3 dz = glm::vec3(0.0f, hz1 - hz0, 2.0f*stepZ);
    glm::vec3 n = glm::normalize(glm::cross(dz, dx));
    if (!std::isfinite(n.x)) n = glm::vec3(0,1,0);
    return n;
}

// ---------------- 序列化 ----------------
static void from_json(const json& j, TerrainSettings& s){
    if (j.contains("type")) s.type = j["type"].get<std::string>();
    if (j.contains("sizeX")) s.sizeX = j["sizeX"].get<float>();
    if (j.contains("sizeZ")) s.sizeZ = j["sizeZ"].get<float>();
    if (j.contains("resolution")) s.resolution = j["resolution"].get<int>();
    if (j.contains("origin")) { auto v=j["origin"]; s.origin={v[0],v[1]}; }
    if (j.contains("noiseAmplitude")) s.noiseAmplitude=j["noiseAmplitude"].get<float>();
    if (j.contains("noiseFrequency")) s.noiseFrequency=j["noiseFrequency"].get<float>();
    if (j.contains("heightmapPath")) s.heightmapPath=j["heightmapPath"].get<std::string>();
    if (j.contains("heightScale"))   s.heightScale=j["heightScale"].get<float>();
    if (j.contains("heightmapFlipY")) s.heightmapFlipY=j["heightmapFlipY"].get<bool>();
    if (j.contains("baseColor"))     { auto v=j["baseColor"]; s.baseColor={v[0],v[1],v[2]}; }
    if (j.contains("wireframe"))     s.wireframe=j["wireframe"].get<bool>();
}

static json to_json(const TerrainSettings& s){
    return json{
        {"type", s.type},
        {"sizeX", s.sizeX},
        {"sizeZ", s.sizeZ},
        {"resolution", s.resolution},
        {"origin", {s.origin.x, s.origin.y}},
        {"noiseAmplitude", s.noiseAmplitude},
        {"noiseFrequency", s.noiseFrequency},
        {"heightmapPath", s.heightmapPath},
        {"heightScale", s.heightScale},
        {"heightmapFlipY", s.heightmapFlipY},
        {"baseColor", {s.baseColor.r, s.baseColor.g, s.baseColor.b}},
        {"wireframe", s.wireframe}
    };
}

// ---------------- Terrain 实现 ----------------
Terrain::Terrain() {}
Terrain::~Terrain() {
    if (ebo) glDeleteBuffers(1, &ebo);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
}

void Terrain::useDefaults(){
    cfg = TerrainSettings{};
}

bool Terrain::loadFromJson(const std::string& jsonPath){
    useDefaults();
    std::ifstream ifs(jsonPath);
    if (!ifs) {
        std::cerr << "[Terrain] Cannot open " << jsonPath << " (use defaults)\n";
    } else {
        try { json j; ifs >> j; from_json(j, cfg); }
        catch (const std::exception& e) {
            std::cerr << "[Terrain] Parse error: " << e.what() << " (use defaults)\n";
        }
    }
    return build();
}

bool Terrain::saveToJson(const std::string& jsonPath) const{
    try {
        std::ofstream ofs(jsonPath);
        if (!ofs) return false;
        ofs << to_json(cfg).dump(2);
        return true;
    } catch (...) { return false; }
}

bool Terrain::build(){
    res = std::max(2, cfg.resolution);
    heightData.assign((res+1)*(res+1), 0.0f);

    if (cfg.type == "flat") {
        generateHeightsFlat();
    } else if (cfg.type == "noise") {
        generateHeightsNoise();
    } else if (cfg.type == "heightmap") {
        if (!generateHeightsFromImage()) {
            std::cerr << "[Terrain] heightmap load failed, fallback to flat\n";
            generateHeightsFlat();
        }
    } else {
        std::cerr << "[Terrain] Unknown type: " << cfg.type << ", fallback to flat\n";
        generateHeightsFlat();
    }

    buildMeshFromHeights();
    ready = true;
    return true;
}

void Terrain::generateHeightsFlat(){
    std::fill(heightData.begin(), heightData.end(), 0.0f);
}

void Terrain::generateHeightsNoise(){
    // 将世界坐标映射为噪声域
    for (int iz=0; iz<=res; ++iz){
        for (int ix=0; ix<=res; ++ix){
            float x = ( (float)ix / (float)res - 0.5f ) * cfg.sizeX + cfg.origin.x;
            float z = ( (float)iz / (float)res - 0.5f ) * cfg.sizeZ + cfg.origin.y;
            float n = noise2D(x, z, cfg.noiseFrequency, 4, 0.5f); // [-1,1]
            heightData[idx(ix,iz)] = n * cfg.noiseAmplitude;
        }
    }
}

bool Terrain::generateHeightsFromImage(){
    if (cfg.heightmapPath.empty() || !fileExists(cfg.heightmapPath)) {
        std::cerr << "[Terrain] heightmap not found: " << cfg.heightmapPath << "\n";
        return false;
    }
    int w,h,comp;
    stbi_set_flip_vertically_on_load(cfg.heightmapFlipY);
    unsigned char* data = stbi_load(cfg.heightmapPath.c_str(), &w,&h,&comp, 1);
    if (!data) {
        std::cerr << "[Terrain] stbi_load failed: " << cfg.heightmapPath << "\n";
        return false;
    }
    // 采样缩放到 (res+1)^2
    for (int iz=0; iz<=res; ++iz){
        for (int ix=0; ix<=res; ++ix){
            float u = (float)ix / (float)res;
            float v = (float)iz / (float)res;
            float x = u * (w-1);
            float y = v * (h-1);
            int x0 = (int)std::floor(x);
            int y0 = (int)std::floor(y);
            int x1 = std::min(x0+1, w-1);
            int y1 = std::min(y0+1, h-1);
            float tx = x - x0;
            float ty = y - y0;

            auto at = [&](int X,int Y){ return data[Y*w + X] / 255.0f; };
            float v00 = at(x0,y0), v10 = at(x1,y0), v01 = at(x0,y1), v11 = at(x1,y1);
            float vx0 = lerp(v00, v10, fade(tx));
            float vx1 = lerp(v01, v11, fade(tx));
            float g   = lerp(vx0, vx1, fade(ty)); // [0,1]

            heightData[idx(ix,iz)] = (g - 0.5f) * 2.0f * cfg.heightScale; // [-heightScale, +heightScale]
        }
    }
    stbi_image_free(data);
    return true;
}

void Terrain::buildMeshFromHeights(){
    // 创建 VAO/VBO/EBO（若已存在重建数据）
    if (!vao) glGenVertexArrays(1, &vao);
    if (!vbo) glGenBuffers(1, &vbo);
    if (!ebo) glGenBuffers(1, &ebo);

    vertices.resize((res+1)*(res+1));
    indices.clear();
    indices.reserve(res*res*6);

    float stepX = cfg.sizeX / (float)res;
    float stepZ = cfg.sizeZ / (float)res;

    for (int iz=0; iz<=res; ++iz){
        for (int ix=0; ix<=res; ++ix){
            int id = idx(ix,iz);
            float xWorld = (ix/(float)res - 0.5f)*cfg.sizeX + cfg.origin.x;
            float zWorld = (iz/(float)res - 0.5f)*cfg.sizeZ + cfg.origin.y;
            float yWorld = heightData[id];

            vertices[id].pos = glm::vec3(xWorld, yWorld, zWorld);
            vertices[id].nrm = calcNormal(heightData, res, ix, iz, stepX, stepZ);
            vertices[id].uv  = glm::vec2(ix/(float)res, iz/(float)res);
        }
    }

    for (int iz=0; iz<res; ++iz){
        for (int ix=0; ix<res; ++ix){
            int i0 = idx(ix,   iz);
            int i1 = idx(ix+1, iz);
            int i2 = idx(ix+1, iz+1);
            int i3 = idx(ix,   iz+1);
            indices.push_back(i0); indices.push_back(i1); indices.push_back(i2);
            indices.push_back(i0); indices.push_back(i2); indices.push_back(i3);
        }
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // layout
    glEnableVertexAttribArray(0); // pos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex,pos));
    glEnableVertexAttribArray(1); // nrm
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex,nrm));
    glEnableVertexAttribArray(2); // uv
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex,uv));

    glBindVertexArray(0);
}

float Terrain::heightAtWorld(float x, float z) const{
    if (!ready) return 0.0f;

    // 映射到网格坐标 [0,res]
    float u = ( (x - cfg.origin.x) / cfg.sizeX ) + 0.5f;
    float v = ( (z - cfg.origin.y) / cfg.sizeZ ) + 0.5f;
    float fx = u * (float)res;
    float fz = v * (float)res;

    // 越界：允许轻微外插
    return sampleHeightBilinear(fx, fz);
}

float Terrain::sampleHeightBilinear(float fx, float fz) const{
    int x0 = (int)std::floor(fx);
    int z0 = (int)std::floor(fz);
    int x1 = x0 + 1;
    int z1 = z0 + 1;

    auto clampi = [&](int v){ return std::max(0, std::min(res, v)); };
    x0 = clampi(x0); x1 = clampi(x1);
    z0 = clampi(z0); z1 = clampi(z1);

    float tx = glm::clamp(fx - std::floor(fx), 0.0f, 1.0f);
    float tz = glm::clamp(fz - std::floor(fz), 0.0f, 1.0f);

    float h00 = heightData[idx(x0,z0)];
    float h10 = heightData[idx(x1,z0)];
    float h01 = heightData[idx(x0,z1)];
    float h11 = heightData[idx(x1,z1)];

    float hx0 = lerp(h00, h10, fade(tx));
    float hx1 = lerp(h01, h11, fade(tx));
    float h   = lerp(hx0, hx1, fade(tz));
    return h;
}

void Terrain::draw(const glm::mat4& view, const glm::mat4& proj, const Environment* env){
    if (!ready) return;

    if (cfg.wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    terrainShader.use();
    terrainShader.setMat4("view", view);
    terrainShader.setMat4("projection", proj);
    terrainShader.setVec3("uBaseColor", cfg.baseColor);

    // 把环境光/太阳/雾 参数也塞给地形 shader（与模型保持一致）
    if (env) {
        // 复用 Environment::applyToShader 的约定（手动 set，避免循环依赖）
        // 与 model fragment 同名 uniform：
        // uAmbientColor, uSunDirection, uSunColor, uFogEnabled, uFogColor, uFogDensity
        // 这里不需要 viewPos（不用镜面）
        // 用一个小技巧：建一个临时 Shader& 别名来重用接口（或直接拷贝字段）
        // 为了简洁，这里直接拿 settings：
        const auto& s = env->settings();
        terrainShader.setVec3("uAmbientColor", s.ambientColor);
        terrainShader.setVec3("uSunDirection", s.sunDirection);
        terrainShader.setVec3("uSunColor",     s.sunColor * s.sunIntensity);
        terrainShader.setInt ("uFogEnabled",   s.fogEnabled ? 1 : 0);
        terrainShader.setVec3("uFogColor",     s.fogColor);
        terrainShader.setFloat("uFogDensity",  s.fogDensity);
    }

    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    if (cfg.wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

