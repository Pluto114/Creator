#version 330 core
in vec3 vPos;
in vec3 vNrm;
out vec4 FragColor;

// 统一使用与 model fragment 相同的环境 uniform
uniform vec3  uAmbientColor;
uniform vec3  uSunDirection; // 从光源指向地面
uniform vec3  uSunColor;
uniform int   uFogEnabled;
uniform vec3  uFogColor;
uniform float uFogDensity;

// 简易固有色
uniform vec3  uBaseColor;

void main(){
    vec3 N = normalize(vNrm);
    vec3 L = normalize(-uSunDirection);

    float ndotl = max(dot(N,L), 0.0);
    vec3 diffuse = uSunColor * ndotl * uBaseColor;
    vec3 ambient = uAmbientColor * uBaseColor;
    vec3 color = ambient + diffuse;

    // 简单雾（exp2）
    if (uFogEnabled == 1) {
        // 由于没有 viewPos，这里用到远近可根据 gl_FragCoord.z 换算；
        // 为简易起见，给一个估计：用世界空间高度近似距离的权重不准确，但足以预览。
        // 若要精准，传 viewPos 并计算 length(viewPos - vPos)。
        float dist = length(vPos);
        float d = uFogDensity * dist;
        float fogFactor = exp(-d*d);
        fogFactor = clamp(fogFactor, 0.0, 1.0);
        color = mix(uFogColor, color, fogFactor);
    }

    FragColor = vec4(color, 1.0);
}
