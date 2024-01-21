#pragma once

#include <cuda.h>
#include <iostream>

namespace utils
{
static constexpr float pi = 3.1415926535897932385f;

__host__ static inline void cuda_check(const cudaError_t error)
{
    if (error != cudaError_t::cudaSuccess)
    {
        std::cout << "ERROR!\n";
        exit(-1);
    }
}

__host__ __device__ static inline float degree_to_radians(const float deg)
{
    return (pi / 180.0f) * deg;
}
} // namespace utils