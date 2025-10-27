#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "Shader.h"

// 地形设置
struct TerrainSettings {
    // "flat" | "noise" | "heightmap"
    std::string type = "flat";

    // 世界尺寸（长宽，单位与场景一致，Y=高度）
    float sizeX = 200.0f;
    float sizeZ = 200.0f;

    // 网格分辨率（顶点数 = (resolution+1)^2）
    int   resolution = 256;

    // 世界原点处的偏移（地形中心相对世界原点）
    glm::vec2 origin = glm::vec2(0.0f, 0.0f);

    // 噪声参数（当 type=noise 时生效）
    float noiseAmplitude = 5.0f;   // 高度振幅（米）
    float noiseFrequency = 0.01f;  // 频率（越大起伏越密）

    // 高度图参数（当 type=heightmap 时生效）
    std::string heightmapPath = "";
    float heightScale = 20.0f;     // 灰度→高度的比例
    bool  heightmapFlipY = true;   // 是否颠倒 Y（常见高度图需要翻转）

    // 外观
    glm::vec3 baseColor = glm::vec3(0.25f, 0.28f, 0.30f);
    bool wireframe = false;
};

class Environment; // 前向声明，便于把环境参数传给地形着色器

class Terrain {
public:
    Terrain();
    ~Terrain();

    // 配置
    bool loadFromJson(const std::string& jsonPath); // 读不到就 defaults
    bool saveToJson(const std::string& jsonPath) const;
    void useDefaults();

    // 构建网格 & CPU 高度场（在 loadFromJson 后自动调用）
    bool build();

    // 渲染（需要传入 VP；内部使用自己的 terrain shader）
    void draw(const glm::mat4& view, const glm::mat4& proj, const Environment* env = nullptr);

    // 世界坐标高度查询（供“贴地”用）
    // x,z 为世界坐标；返回世界高度 y
    float heightAtWorld(float x, float z) const;

    // 访问设置（给 UI 或外部代码改参数再 build）
    TerrainSettings& settings() { return cfg; }
    const TerrainSettings& settings() const { return cfg; }

private:
    // 生成高度数据（按 type）
    void generateHeightsFlat();
    void generateHeightsNoise();
    bool generateHeightsFromImage();

    // 根据 heightData 构建网格（顶点、法线、索引）
    void buildMeshFromHeights();

    // 采样高度（以网格索引空间），双线性插值
    float sampleHeightBilinear(float fx, float fz) const;

    // 工具
    inline int idx(int ix, int iz) const { return iz * (res+1) + ix; }

private:
    TerrainSettings cfg;

    // CPU 高度场：尺寸 (res+1) x (res+1)
    int res = 0;
    std::vector<float> heightData;   // 行主序：z 行，x 列

    // 顶点缓存
    struct Vertex { glm::vec3 pos; glm::vec3 nrm; glm::vec2 uv; };
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    // GL 对象
    unsigned int vao=0, vbo=0, ebo=0;
    Shader terrainShader = Shader("../shaders/terrain_vertex.glsl", "../shaders/terrain_fragment.glsl");
    bool ready = false;
};

