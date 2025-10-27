#version 330 core
out vec4 FragColor;

in vec3 FragPos;    // 世界空间位置（来自 VS）
in vec3 Normal;     // 世界空间法线（来自 VS）
in vec2 TexCoords;  // 纹理坐标（来自 VS）

// 纹理（若未绑定会取到 0）
uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular1;

// 旧版点光参数（保留）
uniform vec3 lightPos;     // 点光源位置（世界）
uniform vec3 viewPos;      // 观察者位置（世界）
uniform vec3 lightColor;   // 点光颜色

// 环境系统参数（由 C++ Environment 传入）
uniform vec3  uAmbientColor;   // 环境光颜色（建议 0.05~0.25）
uniform vec3  uSunDirection;   // 方向光：从光源指向地面的方向（例如 [-0.3,-1,-0.2]）
uniform vec3  uSunColor;       // 方向光颜色 *强度（C++ 已乘 sunIntensity）
uniform int   uFogEnabled;     // 1 开启雾
uniform vec3  uFogColor;       // 雾颜色
uniform float uFogDensity;     // 雾密度（如 0.02）

// ====== 新增：高亮（由 C++ 在 Scene::draw 里设置）======
// 0:无高亮  1:hover  2:selected
uniform int uHighlightState;

// 内置常量色（也可在片元里换成 uniform，自行暴露到 C++）：
const vec3 HOVER_COLOR   = vec3(0.98, 0.85, 0.20); // 金黄
const vec3 SELECT_COLOR  = vec3(0.20, 0.60, 1.00); // 青蓝
const float HOVER_GAIN   = 0.30;                   // 悬停整体增亮
const float SELECT_GAIN  = 0.45;                   // 选中整体增亮
const float RIM_POWER_H  = 3.0;                    // 悬停描边指数
const float RIM_POWER_S  = 2.2;                    // 选中描边指数
const float RIM_SCALE_H  = 0.65;                   // 悬停描边强度
const float RIM_SCALE_S  = 1.00;                   // 选中描边强度

void main()
{
    // ===== 采样 & 基础量 =====
    vec4 albedoTex = texture(texture_diffuse1, TexCoords);
    vec3 albedo    = albedoTex.rgb;

    vec3 N = normalize(Normal);
    vec3 V = normalize(viewPos - FragPos);

    // ===================== 原有点光（保留） =====================
    vec3 Lp = normalize(lightPos - FragPos);   // 点光方向（片元 -> 光）
    vec3 R  = reflect(-Lp, N);

    float ambientStrength = 0.1;
    vec3  ambientLegacy   = ambientStrength * lightColor * albedo;

    float diffP = max(dot(N, Lp), 0.0);
    vec3  diffuseLegacy = diffP * lightColor * albedo;

    float specularStrength = 0.5;
    float specP = pow(max(dot(V, R), 0.0), 32.0);
    vec3  specularLegacy = specularStrength * specP * lightColor;

    // ===================== 新增：方向光（太阳） =====================
    // 方向光方向：从片元指向光源，因此要取 -uSunDirection
    vec3 Ls = normalize(-uSunDirection);
    float ndotl = max(dot(N, Ls), 0.0);

    // 漫反射 & Blinn 高光
    vec3 sunDiffuse = uSunColor * ndotl * albedo;

    vec3 sunSpec = vec3(0.0);
    if (ndotl > 0.0) {
        vec3 H = normalize(Ls + V);                 // Blinn 半程向量
        float shin = 32.0;                          // 可按材质调整
        sunSpec = uSunColor * pow(max(dot(N, H), 0.0), shin);
    }

    // ===================== 新增：环境光 =====================
    vec3 ambientEnv = uAmbientColor * albedo;

    // 合并：旧点光 + 新方向光 + 环境光
    vec3 result = ambientLegacy + ambientEnv
    + diffuseLegacy + specularLegacy
    + sunDiffuse + sunSpec;

    // ===================== 高亮（悬停/选中） =====================
    if (uHighlightState != 0) {
        // 基于视角的“边缘光”作为描边（不需几何/后处理）
        float ndotv = max(dot(N, V), 0.0);
        float rim    = 1.0 - ndotv;                 // 视角越掠，rim 越强

        if (uHighlightState == 1) {
            // Hover：整体稍增亮 + 金黄描边
            float gain = HOVER_GAIN;
            float rimW = pow(rim, RIM_POWER_H) * RIM_SCALE_H;
            result = result * (1.0 + gain) + HOVER_COLOR * rimW;
        } else { // uHighlightState == 2
            // Selected：更亮 + 青蓝描边（偏“自发光”效果）
            float gain = SELECT_GAIN;
            float rimW = pow(rim, RIM_POWER_S) * RIM_SCALE_S;
            result = result * (1.0 + gain) + SELECT_COLOR * rimW;
        }
    }

    // ===================== 新增：指数雾（exp2） =====================
    if (uFogEnabled == 1) {
        float dist = length(viewPos - FragPos);
        float d = uFogDensity * dist;
        float fogFactor = exp(-d * d);             // exp2 雾
        fogFactor = clamp(fogFactor, 0.0, 1.0);
        result = mix(uFogColor, result, fogFactor);
    }

    FragColor = vec4(result, 1.0);
}
