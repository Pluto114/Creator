#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 vDir;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    vDir = aPos;
    vec4 pos = projection * view * vec4(aPos, 1.0);
    gl_Position = pos.xyww; // 深度=1，保证在远平面
}
