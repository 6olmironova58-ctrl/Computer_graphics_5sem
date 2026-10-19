#pragma once
#include <cmath>

namespace math {

// Матрица 4x4 для 3D-преобразований
struct Mat4 {
    float m[4][4] = {};
    
    float* operator[](size_t i) { return m[i]; }
    const float* operator[](size_t i) const { return m[i]; }
};

// Матрица перспективной проекции
inline Mat4 mat4Perspective(float fov, float aspect, float near, float far) {
    Mat4 res;
    float tanHalfFov = std::tan(fov / 2.0f);
    res[0][0] = 1.0f / (aspect * tanHalfFov);
    res[1][1] = 1.0f / tanHalfFov;
    res[2][2] = -(far + near) / (far - near);
    res[2][3] = -(2.0f * far * near) / (far - near);
    res[3][2] = -1.0f;
    return res;
}

// Матрица ортографической проекции
inline Mat4 mat4Orthographic(float left, float right, float bottom, float top, float near, float far) {
    Mat4 res;
    res[0][0] = 2.0f / (right - left);
    res[1][1] = 2.0f / (top - bottom);
    res[2][2] = -2.0f / (far - near);
    res[3][0] = -(right + left) / (right - left);
    res[3][1] = -(top + bottom) / (top - bottom);
    res[3][2] = -(far + near) / (far - near);
    res[3][3] = 1.0f;
    return res;
}

// 2. Матрица вида
inline Mat4 mat4LookAt() {
    Mat4 res;
    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 4; ++j) {
            res[i][j] = 0.0f;
        }
    }
    res[0][0] = 1.0f; 
    res[1][1] = 1.0f; 
    res[2][2] = 1.0f; 
    res[3][3] = 1.0f;
    
    res[2][3] = -3.0f; 
    return res;
}

// 3. Матрица модели
inline Mat4 mat4RotateY(float angle) {
    Mat4 res;
    res[0][0] = std::cos(angle); res[0][2] = std::sin(angle);
    res[2][0] = -std::sin(angle); res[2][2] = std::cos(angle);
    res[1][1] = 1.0f; 
    res[3][3] = 1.0f;
    return res;
}

// Перемещение (Translation)
inline Mat4 mat4Translate(float x, float y, float z) {
    Mat4 res;
    res[0][0] = 1.0f; res[1][1] = 1.0f; res[2][2] = 1.0f; res[3][3] = 1.0f;
    res[3][0] = x; res[3][1] = y; res[3][2] = z;
    return res;
}

// Вращение вокруг оси X
inline Mat4 mat4RotateX(float angle) {
    Mat4 res;
    res[0][0] = 1.0f; res[3][3] = 1.0f;
    res[1][1] = std::cos(angle); res[1][2] = -std::sin(angle);
    res[2][1] = std::sin(angle); res[2][2] = std::cos(angle);
    return res;
}

// Вращение вокруг оси Z
inline Mat4 mat4RotateZ(float angle) {
    Mat4 res;
    res[2][2] = 1.0f; res[3][3] = 1.0f;
    res[0][0] = std::cos(angle); res[0][1] = -std::sin(angle);
    res[1][0] = std::sin(angle); res[1][1] = std::cos(angle);
    return res;
}

// Масштабирование (Scaling)
inline Mat4 mat4Scale(float x, float y, float z) {
    Mat4 res;
    res[0][0] = x; res[1][1] = y; res[2][2] = z; res[3][3] = 1.0f;
    return res;
}

// Умножение двух матриц 4x4
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

} // namespace math