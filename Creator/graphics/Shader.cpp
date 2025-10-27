#include "Shader.h"

#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <iostream>

std::string Shader::loadTextFile(const char* path)
{
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        std::ostringstream oss;
        oss << "[Shader] Cannot open file: " << path;
        throw std::runtime_error(oss.str());
    }
    std::ostringstream ss; ss << ifs.rdbuf();
    return ss.str();
}

Shader::Shader(const char* vertexPath, const char* fragmentPath)
{
    const std::string vsrc = loadTextFile(vertexPath);
    const std::string fsrc = loadTextFile(fragmentPath);

    const char* vcode = vsrc.c_str();
    const char* fcode = fsrc.c_str();

    unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vcode, nullptr);
    glCompileShader(vs);
    checkCompileErrors(vs, "VERTEX");

    unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fcode, nullptr);
    glCompileShader(fs);
    checkCompileErrors(fs, "FRAGMENT");

    ID = glCreateProgram();
    glAttachShader(ID, vs);
    glAttachShader(ID, fs);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");

    glDeleteShader(vs);
    glDeleteShader(fs);
}

Shader::~Shader()
{
    if (ID) {
        glDeleteProgram(ID);
        ID = 0;
    }
}

void Shader::use() const
{
    glUseProgram(ID);
}

void Shader::setBool(const std::string& name, bool value) const
{
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value ? 1 : 0);
}
void Shader::setInt(const std::string& name, int value) const
{
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}
void Shader::setFloat(const std::string& name, float value) const
{
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::setVec2(const std::string& name, const glm::vec2& value) const
{
    glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(value));
}
void Shader::setVec2(const std::string& name, float x, float y) const
{
    glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
}

void Shader::setVec3(const std::string& name, const glm::vec3& value) const
{
    glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(value));
}
void Shader::setVec3(const std::string& name, float x, float y, float z) const
{
    glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
}

void Shader::setVec4(const std::string& name, const glm::vec4& value) const
{
    glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(value));
}
void Shader::setVec4(const std::string& name, float x, float y, float z, float w) const
{
    glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
}

void Shader::setMat2(const std::string& name, const glm::mat2& mat) const
{
    glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}
void Shader::setMat3(const std::string& name, const glm::mat3& mat) const
{
    glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}
void Shader::setMat4(const std::string& name, const glm::mat4& mat) const
{
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}

void Shader::checkCompileErrors(unsigned int object, const char* type) const
{
    GLint success = 0;
    if (std::string(type) == "PROGRAM") {
        glGetProgramiv(object, GL_LINK_STATUS, &success);
        if (!success) {
            GLint len = 0; glGetProgramiv(object, GL_INFO_LOG_LENGTH, &len);
            std::vector<char> log(len > 0 ? len : 1);
            glGetProgramInfoLog(object, (GLsizei)log.size(), nullptr, log.data());
            std::cerr << "[Shader] Link failed:\n" << log.data() << std::endl;
        }
    } else {
        glGetShaderiv(object, GL_COMPILE_STATUS, &success);
        if (!success) {
            GLint len = 0; glGetShaderiv(object, GL_INFO_LOG_LENGTH, &len);
            std::vector<char> log(len > 0 ? len : 1);
            glGetShaderInfoLog(object, (GLsizei)log.size(), nullptr, log.data());
            std::cerr << "[Shader] Compile failed (" << type << "):\n"
                      << log.data() << std::endl;
        }
    }
}
