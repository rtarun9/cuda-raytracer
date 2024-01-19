#pragma once

#include <random>
#include "math/float3.hpp"

namespace utils
{
    static inline void cuda_check(const cudaError_t error)
    {
        if (error != cudaError_t::cudaSuccess)
        {
            std::cout << "ERROR!\n";
        }
    }

    __host__ __device__ static inline float get_random_float_in_range_0_1()
    {
        static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        static std::mt19937 generator{};

        return distribution(generator);
    }

    __host__ __device__ static inline float get_random_float_in_range(const float min, const float max)
    {
        return min + (max - min) * get_random_float_in_range_0_1();
    }

    static constexpr float pi = 3.1415926535897932385f;

    __host__ __device__ static inline float degree_to_radians(const float deg)
    {
        return (pi / 180.0f) * deg;
    }

    __host__ __device__ static math::float3 get_random_float3_in_range(const float min, const float max)
    {
        return math::float3(get_random_float_in_range(min, max), get_random_float_in_range(min, max), get_random_float_in_range(min, max));
    }

    __host__ __device__ static math::float3 get_float3_in_cube()
    {
        return math::float3(get_random_float_in_range(-1.0f, 1.0f), get_random_float_in_range(-1.0f, 1.0f), get_random_float_in_range(-1.0f, 1.0f));
    }

    __host__ __device__ static math::float3 get_random_float3_in_sphere()
    {
        // To get a random float3 in sphere, use the rejection method.
        // First get random float3 in cube (with w / h / d in range -1 to 1).
        // Then, if the length of the float3 is <= 1.0f, it is in the sphere.
        while (true)
        {
            const auto random_float3 = get_float3_in_cube();
            const auto mag = random_float3.magnitude();

            if (mag <= 1.0f)
            {
                return random_float3;
            }
        }
    }

    __host__ __device__ static math::float3 get_random_float3_in_disk()
    {
        while (true)
        {
            const auto random_float3 = math::float3(get_random_float_in_range(-1.0f, 1.0f), get_random_float_in_range(-1.0f, 1.0f), 0.0f);

            if (random_float3.len() <= 1.0f)
            {
                return random_float3;
            }
        }
    }

    __host__ __device__ static math::float3 get_random_unit_float3()
    {
        // If you get random float3 in cube, note that the range of values
        // is [-1, -1, -1] to [1, 1, 1].
        // The dist between diagonal corner and other corners is not the same.
        // This can lead to potentially non - uniform distribution, which is what we want to avoid since lot of float3's will be along the (longest) diagonal in the cube.
        // For that reason we first get random float3 in sphere, and then normalize it.
        // https://github.com/RayTracing/raytracing.github.io/discussions/941
        return get_random_float3_in_sphere().normalize();
    }
}