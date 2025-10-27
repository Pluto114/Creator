#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec2 aUV;

out vec3 vPos;
out vec3 vNrm;

uniform mat4 view;
uniform mat4 projection;

void main(){
    vPos = aPos;
    vNrm = aNrm;
    gl_Position = projection * view * vec4(aPos, 1.0);
}
