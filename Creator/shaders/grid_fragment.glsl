#version 330 core
in vec3 WorldPos_FS; // 从顶点着色器接收到的世界坐标
out vec4 FragColor;

uniform vec3 gridColor;         // 网格线颜色
uniform vec3 cameraPosition;    // 摄像机世界位置
uniform float majorGridSpacing; // 主网格间距
uniform float minorGridSpacing; // 子网格间距
uniform float fadeStartDistance; // 网格开始淡出的距离
uniform float fadeEndDistance;   // 网格完全淡出的距离

// 使用 fwidth 实现抗锯齿线条的函数
float smoothedLine(float value, float spacing, float thicknessRelativeToSpacing) {
    float halfSpacing = spacing * 0.5;
    float wrappedValue = mod(value + halfSpacing, spacing) - halfSpacing;
    float distToLine = abs(wrappedValue);
    float lineThickness = fwidth(value) * thicknessRelativeToSpacing * 2.0;
    return smoothstep(lineThickness, 0.0, distToLine);
}

void main() {
    // 1. 计算所有网格线（主、次、轴）的组合透明度
    float lineAlpha = 0.0;

    // 子网格线
    float minorLines = smoothedLine(WorldPos_FS.x, minorGridSpacing, 0.5) +
    smoothedLine(WorldPos_FS.z, minorGridSpacing, 0.5);
    lineAlpha = max(lineAlpha, clamp(minorLines, 0.0, 1.0) * 0.3);

    // 主网格线
    float majorLines = smoothedLine(WorldPos_FS.x, majorGridSpacing, 1.0) +
    smoothedLine(WorldPos_FS.z, majorGridSpacing, 1.0);
    lineAlpha = max(lineAlpha, clamp(majorLines, 0.0, 1.0) * 0.7);

    // XZ 轴
    float axisThicknessFactor = 2.0;
    if (abs(WorldPos_FS.z) < fwidth(WorldPos_FS.z) * axisThicknessFactor) { // X轴
        lineAlpha = max(lineAlpha, smoothedLine(WorldPos_FS.x, majorGridSpacing, axisThicknessFactor * 0.5));
    }
    if (abs(WorldPos_FS.x) < fwidth(WorldPos_FS.x) * axisThicknessFactor) { // Z轴
        lineAlpha = max(lineAlpha, smoothedLine(WorldPos_FS.z, majorGridSpacing, axisThicknessFactor * 0.5));
    }

    // 如果没有任何线，lineAlpha 会非常接近 0

    // 2. 计算距离淡出效果
    float distToCamPlane = length(WorldPos_FS.xz - cameraPosition.xz);
    float fadeAmount = smoothstep(fadeStartDistance, fadeEndDistance, distToCamPlane);

    // 3. 计算最终透明度
    // 基础透明度是 lineAlpha，然后根据距离淡出
    float finalAlpha = lineAlpha * (1.0 - fadeAmount);

    // 4. 输出最终颜色
    // RGB 通道始终是网格线颜色
    // Alpha 通道是我们计算出的最终透明度
    // 如果 finalAlpha 为 0，这个片段就是完全透明的
    FragColor = vec4(gridColor, finalAlpha);
}