#include "Camera.h" // 包含对应的头文件
#include <glm/gtc/matrix_transform.hpp> // 包含 lookAt 等函数
#include <iostream> // 用于可能的调试输出



// 构造函数 (向量版本)
Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) :
    Front(glm::vec3(0.0f, 0.0f, -1.0f)), // 初始 Front 指向 -Z
    MovementSpeed(SPEED),
    MouseSensitivity(SENSITIVITY),
    Zoom(ZOOM)
{
    Position = position;
    WorldUp = up;
    Yaw = yaw;
    Pitch = pitch;
    updateCameraVectors(); // 初始化时计算 Front, Right, Up 向量
}

// 构造函数 (标量版本)
Camera::Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) :
    Front(glm::vec3(0.0f, 0.0f, -1.0f)),
    MovementSpeed(SPEED),
    MouseSensitivity(SENSITIVITY),
    Zoom(ZOOM)
{
    Position = glm::vec3(posX, posY, posZ);
    WorldUp = glm::vec3(upX, upY, upZ);
    Yaw = yaw;
    Pitch = pitch;
    updateCameraVectors();
}

// 获取视图矩阵
glm::mat4 Camera::GetViewMatrix() {
    // 使用 glm::lookAt 计算视图矩阵
    // 参数: 摄像机位置, 目标位置 (当前位置 + 前向量), 上向量
    return glm::lookAt(Position, Position + Front, Up);
}

// 处理键盘输入
void Camera::ProcessKeyboard(Camera_Movement direction, float deltaTime) {
    float velocity = MovementSpeed * deltaTime; // 根据帧时间调整移动速度，实现平滑移动
    if (direction == FORWARD)
        Position += Front * velocity;
    if (direction == BACKWARD)
        Position -= Front * velocity;
    if (direction == LEFT)
        Position -= Right * velocity; // 注意是减去 Right 向量
    if (direction == RIGHT)
        Position += Right * velocity; // 注意是加上 Right 向量
    if (direction == UP)
        Position += WorldUp * velocity; // 使用世界 Up 进行上下移动
    if (direction == DOWN)
        Position -= WorldUp * velocity;
    // 可以选择是否固定 Y 轴 (伪飞行模式)，如果需要 FPS 模式，可以注释掉上下移动或只改变 XZ
    // Position.y = 0.0f; // 如果想让摄像机固定在 XZ 平面
}

// 处理鼠标移动
void Camera::ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch) {
    // 用鼠标灵敏度调整偏移量
    xoffset *= MouseSensitivity;
    yoffset *= MouseSensitivity;

    Yaw   += xoffset;
    Pitch += yoffset; // 注意这里是加号，如果需要反转 Y 轴则用减号

    // 限制俯仰角范围，防止万向锁和视角颠倒
    if (constrainPitch) {
        if (Pitch > 89.0f)
            Pitch = 89.0f;
        if (Pitch < -89.0f)
            Pitch = -89.0f;
    }

    // 更新 Front, Right 和 Up 向量
    updateCameraVectors();
}

// 处理鼠标滚轮 (可选)
void Camera::ProcessMouseScroll(float yoffset) {
    Zoom -= (float)yoffset; // 滚轮向前 Zoom 减小 (视野变窄，放大效果)
    // 限制 Zoom 范围
    if (Zoom < 1.0f)
        Zoom = 1.0f;
    if (Zoom > 45.0f) // 或者你想要的最大视野
        Zoom = 45.0f;
}

// 私有方法：根据 Yaw 和 Pitch 计算内部向量
void Camera::updateCameraVectors() {
    // 计算新的 Front 向量
    glm::vec3 front;
    front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    front.y = sin(glm::radians(Pitch));
    front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    Front = glm::normalize(front); // 标准化向量
    // 重新计算 Right 和 Up 向量
    Right = glm::normalize(glm::cross(Front, WorldUp)); // 右向量 = 前向量 X 世界Up向量
    Up    = glm::normalize(glm::cross(Right, Front));   // 上向量 = 右向量 X 前向量
}
