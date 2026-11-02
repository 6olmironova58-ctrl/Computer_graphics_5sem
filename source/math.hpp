#pragma once
#include <cmath>
#include <cstddef>

namespace math {

constexpr float kPi = 3.14159265358979f;

inline float radians(float degrees) { return degrees * kPi / 180.0f; }

struct Mat4 {
    float m[4][4] = {};

    float* operator[](size_t i) { return m[i]; }
    const float* operator[](size_t i) const { return m[i]; }
};

inline Mat4 mat4Identity() {
    Mat4 res;
    res[0][0] = 1.0f; res[1][1] = 1.0f; res[2][2] = 1.0f; res[3][3] = 1.0f;
    return res;
}

inline Mat4 mat4Perspective(float fov, float aspect, float zNear, float zFar) {
    Mat4 res;
    float f = 1.0f / std::tan(fov / 2.0f);
    res[0][0] = f / aspect;
    res[1][1] = f;
    res[2][2] = zFar / (zNear - zFar);
    res[2][3] = (zFar * zNear) / (zNear - zFar);
    res[3][2] = -1.0f;
    return res;
}

inline Mat4 mat4Orthographic(float left, float right, float bottom, float top, float zNear, float zFar) {
    Mat4 res;
    res[0][0] = 2.0f / (right - left);
    res[0][3] = -(right + left) / (right - left);
    res[1][1] = 2.0f / (top - bottom);
    res[1][3] = -(top + bottom) / (top - bottom);
    res[2][2] = -1.0f / (zFar - zNear);
    res[2][3] = -zNear / (zFar - zNear);
    res[3][3] = 1.0f;
    return res;
}

inline Mat4 mat4Translate(float x, float y, float z) {
    Mat4 res = mat4Identity();
    res[0][3] = x; res[1][3] = y; res[2][3] = z;
    return res;
}

inline Mat4 mat4RotateX(float angle) {
    Mat4 res;
    res[0][0] = 1.0f; res[3][3] = 1.0f;
    res[1][1] = std::cos(angle); res[1][2] = -std::sin(angle);
    res[2][1] = std::sin(angle); res[2][2] = std::cos(angle);
    return res;
}

inline Mat4 mat4RotateY(float angle) {
    Mat4 res;
    res[1][1] = 1.0f; res[3][3] = 1.0f;
    res[0][0] = std::cos(angle); res[0][2] = std::sin(angle);
    res[2][0] = -std::sin(angle); res[2][2] = std::cos(angle);
    return res;
}

inline Mat4 mat4RotateZ(float angle) {
    Mat4 res;
    res[2][2] = 1.0f; res[3][3] = 1.0f;
    res[0][0] = std::cos(angle); res[0][1] = -std::sin(angle);
    res[1][0] = std::sin(angle); res[1][1] = std::cos(angle);
    return res;
}

inline Mat4 mat4Scale(float x, float y, float z) {
    Mat4 res;
    res[0][0] = x; res[1][1] = y; res[2][2] = z; res[3][3] = 1.0f;
    return res;
}

inline Mat4 mat4Multiply(const Mat4& a, const Mat4& b) {
    Mat4 res;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
}

// Матрица вида = обратная к матрице модели камеры C = T * Ry * Rx * Rz,
inline Mat4 mat4View(const float position[3], float pitch, float yaw, float roll) {
    Mat4 rotation = mat4Multiply(mat4RotateZ(-roll),
                    mat4Multiply(mat4RotateX(-pitch), mat4RotateY(-yaw)));
    Mat4 translation = mat4Translate(-position[0], -position[1], -position[2]);
    return mat4Multiply(rotation, translation);
}

} // namespace math
