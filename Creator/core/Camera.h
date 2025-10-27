#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

// 定义摄像机可能的移动方向，作为输入抽象
enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,      // 可选：向上移动
    DOWN     // 可选：向下移动
};

// 默认摄像机参数值
const float YAW         = -90.0f; // 偏航角，-90度指向 Z 负轴 (OpenGL前方)
const float PITCH       =  0.0f;  // 俯仰角
const float SPEED       =  2.5f;  // 移动速度
const float SENSITIVITY =  0.1f;  // 鼠标灵敏度
const float ZOOM        =  45.0f; // 视野 (Field of View)

// 一个抽象的摄像机类，用于处理输入并计算相应的欧拉角、向量和视图矩阵
class Camera {
public:
    // 摄像机属性
    glm::vec3 Position; // 位置
    glm::vec3 Front;    // 朝向（前方）
    glm::vec3 Up;       // 上向量 (局部坐标系)
    glm::vec3 Right;    // 右向量 (局部坐标系)
    glm::vec3 WorldUp;  // 世界空间的上向量 (通常是 0,1,0)
    // 欧拉角 (角度)
    float Yaw;
    float Pitch;
    // 摄像机选项
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom; // 用于透视投影的视野角度

    // --- 构造函数 ---
    // 使用向量初始化
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 3.0f), // 默认位置在 Z 轴正方向 3 个单位
           glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),      // 默认世界 Up
           float yaw = YAW, float pitch = PITCH);
    // 使用标量值初始化
    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);

    // --- 公有方法 ---
    // 返回使用欧拉角和 LookAt 矩阵计算出的视图矩阵
    glm::mat4 GetViewMatrix();

    // 处理从键盘接收到的输入
    void ProcessKeyboard(Camera_Movement direction, float deltaTime);

    // 处理从鼠标输入系统接收到的偏移量
    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true);

    // 处理鼠标滚轮事件 (可选，用于缩放视野)
    void ProcessMouseScroll(float yoffset);

private:
    // 根据摄像机的(更新后的)欧拉角计算前向量、右向量和上向量
    void updateCameraVectors();



};

#endif // CAMERA_H
