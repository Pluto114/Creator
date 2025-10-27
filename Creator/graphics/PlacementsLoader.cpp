// graphics/PlacementsLoader.cpp  (最终版：优先使用统一尺度 registry + 多格式兼容)

#include "PlacementsLoader.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <cstdlib> // for std::getenv
#include <algorithm>
#include <ObjectArray.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Scene.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

// -------------------------- 工具与辅助 --------------------------

static bool FileExists(const std::string& p){
    std::error_code ec;
    return fs::exists(p, ec) && fs::is_regular_file(p, ec);
}

static std::string JoinPath(const std::string& a, const std::string& b){
#ifdef _WIN32
    const char sep = '\\';
#else
    const char sep = '/';
#endif
    if(a.empty()) return b;
    if(b.empty()) return a;
    if(a.back()==sep) return a + b;
    return a + sep + b;
}

static std::string Dirname(const std::string& p){
    std::error_code ec;
    fs::path ph(p);
    return ph.has_parent_path() ? ph.parent_path().string() : std::string();
}

// SDL → id->category
static std::unordered_map<std::string,std::string>
LoadCategoriesFromSDL(const std::string& sdlPath) {
    std::unordered_map<std::string,std::string> m;
    if (sdlPath.empty()) return m;
    std::ifstream ifs(sdlPath);
    if (!ifs) return m;
    try {
        json j; ifs >> j;
        if (j.contains("objects") && j["objects"].is_array()) {
            for (auto& o : j["objects"]) {
                if (o.contains("id") && o.contains("category"))
                    m[o["id"].get<std::string>()] = o["category"].get<std::string>();
            }
        } else if (j.contains("categories") && j["categories"].is_object()) {
            for (auto it = j["categories"].begin(); it != j["categories"].end(); ++it) {
                m[it.key()] = it.value().get<std::string>();
            }
        }
    } catch(...) {}
    return m;
}

// 在 dir 中找最新的 scene_*.json（用于补全 category）
static std::string FindLatestSDLBeside(const std::string& dir){
#ifdef _WIN32
    std::string pattern = JoinPath(dir, "scene_*.json");
    WIN32_FIND_DATAA ffd{}; HANDLE h = FindFirstFileA(pattern.c_str(), &ffd);
    if (h == INVALID_HANDLE_VALUE) return "";
    FILETIME latest{0,0}; std::string latestPath;
    do {
        if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            FILETIME ft = ffd.ftLastWriteTime;
            if (CompareFileTime(&latest, &ft) < 0) {
                latest = ft; latestPath = JoinPath(dir, ffd.cFileName);
            }
        }
    } while (FindNextFileA(h, &ffd));
    FindClose(h); return latestPath;
#else
    (void)dir; return "";
#endif
}

// 解析字符串到绝对路径（优先基于 registryDir，再基于 assetsRoot）
static std::string ResolveByBases(const std::string& raw,
                                  const std::string& registryDir,
                                  const std::string& assetsRoot)
{
    if (raw.empty()) return {};
    // 绝对路径
#ifdef _WIN32
    bool isAbs = (raw.size()>=3 && std::isalpha((unsigned char)raw[0]) && raw[1]==':' &&
                  (raw[2]=='\\'||raw[2]=='/')) || (!raw.empty() && (raw[0]=='\\'||raw[0]=='/'));
#else
    bool isAbs = !raw.empty() && raw[0]=='/';
#endif
    if (isAbs && FileExists(raw)) return raw;

    // 基于 registryDir
    if (!registryDir.empty()){
        std::string p = JoinPath(registryDir, raw);
        if (FileExists(p)) return p;
    }
    // 基于 assetsRoot
    if (!assetsRoot.empty()){
        std::string p = JoinPath(assetsRoot, raw);
        if (FileExists(p)) return p;
    }
    return {};
}

// -------------------------- Registry 读取（多格式兼容） --------------------------
//
// 支持：
// 1) {"map": { "id": "path", ... }}
// 2) 扁平 { "id": "path", ... }
// 3) {"models":[ {"id":"...","path":"..."} ... ]}    // 也兼容 asset_path/file/relpath
//
static std::unordered_map<std::string,std::string>
LoadRegistryFlexible(const std::string& registryPath,
                     const std::string& assetsRoot)
{
    std::unordered_map<std::string,std::string> m;
    if (registryPath.empty()) return m;

    std::ifstream ifs(registryPath);
    if(!ifs) return m;

    const std::string baseDir = Dirname(registryPath);

    try{
        json j; ifs >> j;

        auto take_str_val = [&](const std::string& id, const std::string& raw){
            std::string resolved = ResolveByBases(raw, baseDir, assetsRoot);
            if (!resolved.empty()) { m[id]=resolved; return; }
            // 没找到就原样塞进去，后面还有兜底
            m[id]=raw;
        };

        // 3) models 数组
        if (j.contains("models") && j["models"].is_array()){
            for (const auto& e : j["models"]){
                if (!e.is_object()) continue;
                std::string id;
                if (e.contains("id") && e["id"].is_string()) id = e["id"].get<std::string>();
                else if (e.contains("name") && e["name"].is_string()) id = e["name"].get<std::string>();
                else if (e.contains("src_object") && e["src_object"].is_string()) id = e["src_object"].get<std::string>();
                if (id.empty()) continue;

                std::string path;
                if (e.contains("path") && e["path"].is_string()) path = e["path"].get<std::string>();
                else if (e.contains("asset_path") && e["asset_path"].is_string()) path = e["asset_path"].get<std::string>();
                else if (e.contains("file") && e["file"].is_string()) path = e["file"].get<std::string>();
                else if (e.contains("relpath") && e["relpath"].is_string()) path = e["relpath"].get<std::string>();

                if (!path.empty()) {
                    take_str_val(id, path);
                } else {
                    // 没写路径就猜：id.glb 放在 registry 目录或 assetsRoot
                    std::string guess = JoinPath(baseDir, id + ".glb");
                    if (!FileExists(guess) && !assetsRoot.empty()) {
                        std::string guess2 = JoinPath(assetsRoot, id + ".glb");
                        if (FileExists(guess2)) guess = guess2;
                    }
                    if (FileExists(guess)) m[id]=guess;
                }
            }
        }

        // 1) map 对象
        if (j.contains("map") && j["map"].is_object()){
            for(auto it=j["map"].begin(); it!=j["map"].end(); ++it){
                if(it.value().is_string()) take_str_val(it.key(), it.value().get<std::string>());
            }
        }

        // 2) 扁平对象
        if(j.is_object()){
            for(auto it=j.begin(); it!=j.end(); ++it){
                if(it->is_string()){
                    take_str_val(it.key(), it->get<std::string>());
                }
            }
        }

    }catch(const std::exception& e){
        std::cerr << "[PlacementsLoader] Registry parse error: " << e.what() << "\n";
    }

    return m;
}

// -------------------------- 资产路径解析（优先 registry） --------------------------
//
// 查询顺序：
//   (1) registry[id]  → 绝对/相对转绝对（基于 registryDir / assetsRoot）
//   (2) pm.asset_path → 绝对/相对转绝对（基于 assetsRoot）
//   (3) assetsRoot/id.{glb,gltf,obj,fbx,ply,stl}
//   (4) 遍历 assetsRoot 直接匹配同名 stem
//   (5) 占位模型
//
static std::string ResolveAssetPath(const json& pm,
                                    const std::unordered_map<std::string,std::string>& registry,
                                    const std::string& assetsRoot,
                                    const std::string& registryDir,
                                    const std::string& placeholderModel)
{
    std::string src = pm.value("src_object", pm.value("id", std::string()));
    if (src.empty()){
        // 尽量再兜底一下：有些生成器可能只给 "name"
        src = pm.value("name", std::string());
    }

    // 1) registry
    auto it = registry.find(src);
    if (it != registry.end()){
        const std::string& mapped = it->second;
        // 如果 registry 已经是绝对路径并存在，直接用；否则按 baseDir/artsRoot 解析
        if (FileExists(mapped)) return mapped;
        std::string resolved = ResolveByBases(mapped, registryDir, assetsRoot);
        if (FileExists(resolved)) return resolved;

        // 如果仅给了文件名，尝试 registryDir / assetsRoot 拼接
        std::string guess = JoinPath(registryDir, mapped);
        if (FileExists(guess)) return guess;
        if (!assetsRoot.empty()){
            guess = JoinPath(assetsRoot, mapped);
            if (FileExists(guess)) return guess;
        }
        // 再猜 id.glb 放在 registryDir
        std::string guess2 = JoinPath(registryDir, src + ".glb");
        if (FileExists(guess2)) return guess2;
    }

    // 2) models[i].asset_path
    if (pm.contains("asset_path") && pm["asset_path"].is_string()){
        std::string ap = pm["asset_path"].get<std::string>();
        if (FileExists(ap)) return ap;
        if (!assetsRoot.empty()){
            std::string guess = JoinPath(assetsRoot, ap);
            if (FileExists(guess)) return guess;
        }
    }

    // 根目录（环境变量优先）
    std::string root = assetsRoot;
    if(const char* envRoot = std::getenv("CREATOR_ASSETS_ROOT")){
        root = std::string(envRoot);
    }

    // 3) 在根目录中按 stem==id 查找常见扩展
    static const char* exts[] = { ".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl" };
    for(const char* ext: exts){
        std::string p = JoinPath(root, src + ext);
        if(FileExists(p)) return p;
    }

    // 3.1) 兜底：遍历根目录匹配 stem
    std::error_code ec;
    if(fs::exists(root, ec) && fs::is_directory(root, ec)){
        for(auto& entry : fs::directory_iterator(root, ec)){
            if(entry.is_regular_file()){
                if(entry.path().stem().string() == src){
                    return entry.path().string();
                }
            }
        }
    }

    // 4) 占位
    return placeholderModel;
}

// -------------------------- 主入口 --------------------------

bool LoadPlacementsIntoScene(const std::string& placementsJsonPath,
                             Scene& scene,
                             const PlacementLoadOptions& opts)
{
    // 1) 读取 placements.json
    std::ifstream ifs(placementsJsonPath);
    if (!ifs) {
        std::cerr << "[PlacementsLoader] Cannot open " << placementsJsonPath << "\n";
        return false;
    }
    json j;
    try {
        ifs >> j;
    } catch (const std::exception& e) {
        std::cerr << "[PlacementsLoader] JSON parse error: " << e.what() << "\n";
        return false;
    }

    if (!j.contains("models") || !j["models"].is_array()) {
        std::cerr << "[PlacementsLoader] 'models' array missing.\n";
        return false;
    }

    if (opts.clearSceneBeforeLoad) {
        scene.clear();
    }

    // 2) 读取（可选）资产注册表 / 路径映射 —— 优先使用统一尺度输出的 registry
    const std::string registryPath = opts.registryPath;
    const std::string registryDir  = Dirname(registryPath);
    auto registry = LoadRegistryFlexible(registryPath, opts.assetsRoot);

    // 3) 补充类别映射（可选）
    std::unordered_map<std::string, std::string> categoryMap;
    // 3.1 placements 内若有 source_sdl 优先
    std::string sdlPath;
    if (j.contains("source_sdl") && j["source_sdl"].is_string()){
        sdlPath = j["source_sdl"].get<std::string>();
        if (!FileExists(sdlPath)) sdlPath.clear();
    }
    // 3.2 同目录查找最新 scene_*.json 兜底
    if (sdlPath.empty()){
        sdlPath = FindLatestSDLBeside(Dirname(placementsJsonPath));
    }
    if (!sdlPath.empty()){
        categoryMap = LoadCategoriesFromSDL(sdlPath);
    }

    // 4) 遍历并实例化
    int loaded = 0;
    for (const auto& pm : j["models"]) {
        // 4.1 解析 TRS
        glm::vec3 pos(0.0f), rotDeg(0.0f), scl(1.0f);
        try {
            const auto& p = pm.at("position");
            pos = glm::vec3(p.at(0).get<float>(), p.at(1).get<float>(), p.at(2).get<float>());
        } catch (...) {}
        try {
            const auto& r = pm.at("rotation_euler_deg");
            rotDeg = glm::vec3(r.at(0).get<float>(), r.at(1).get<float>(), r.at(2).get<float>());
        } catch (...) {}
        float s = pm.value("scale", 1.0f);
        scl = glm::vec3(s);

        // 4.2 路径解析（优先 registry → 统一尺度后的 glb）
        std::string assetPath = ResolveAssetPath(pm, registry, opts.assetsRoot, registryDir, opts.placeholderModel);
        if (!FileExists(assetPath)) {
            std::cerr << "[PlacementsLoader] Missing asset (even after registry/assetsRoot search): " << assetPath << "\n";
            continue;
        }

        // 4.3 实例名称与来源（src_object）
        const std::string srcId = pm.value("src_object", pm.value("id", std::string()));
        const std::string name  = pm.value("id", srcId.empty() ? "instance" : srcId);
        const std::string cat   = (!srcId.empty() && categoryMap.count(srcId)) ? categoryMap.at(srcId) : std::string();

        // 4.4 加入场景
        int instId = scene.addModel(assetPath, pos, rotDeg, scl, name);
        if (instId >= 0) {
            if (auto* inst = scene.get(instId)) {
                inst->sourceId = srcId;   // 记录来源 id，供贴地/规则使用
                inst->category = cat;     // 记录类别（若解析不到则为空）
            }
            ++loaded;
        } else {
            std::cerr << "[PlacementsLoader] addModel failed: " << assetPath << "\n";
        }
    }

    std::cout << "[PlacementsLoader] Loaded " << loaded
              << " model(s) from " << placementsJsonPath << "\n";
    return loaded > 0;
}
