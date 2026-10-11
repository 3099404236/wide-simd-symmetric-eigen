// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MatrixSymmetricEigenLayoutData {
    uint32_t n;
    uint32_t batch_size;
    uint32_t is_upper;
};