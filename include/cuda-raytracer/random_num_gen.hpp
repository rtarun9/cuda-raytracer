#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include <random>

static constexpr float pi = 3.1415926535897932385f;

__host__ __device__ static inline float get_random_float_in_range_0_1()
{

#ifndef __CUDA_ARCH__
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    static std::mt19937 generator{};

    return distribution(generator);
#else

    // Reference :
    // https://stackoverflow.com/questions/22425283/how-could-we-generate-random-numbers-in-cuda-c-with-different-seed-on-each-run
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(clock64(), i, 0, &state);
    return curand_uniform(&state);
#endif
}

__host__ __device__ static inline float get_random_float_in_range(const float min, const float max)
{
    return min + (max - min) * get_random_float_in_range_0_1();
}

__host__ __device__ static math::float3 get_random_float3_in_range(const float min, const float max)
{
    return math::float3(get_random_float_in_range(min, max), get_random_float_in_range(min, max),
                        get_random_float_in_range(min, max));
}

__host__ __device__ static math::float3 get_float3_in_cube()
{
    return math::float3(get_random_float_in_range(-1.0f, 1.0f), get_random_float_in_range(-1.0f, 1.0f),
                        get_random_float_in_range(-1.0f, 1.0f));
}

__host__ __device__ static math::float3 get_random_float3_in_sphere()
{
    // See here : https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/ for the code.
    float u = get_random_float_in_range_0_1();
    float v = get_random_float_in_range_0_1();

    float theta = u * 2.0f * pi;
    float phi = acos(2.0f * v - 1.0f);

    float r = pow(get_random_float_in_range_0_1(), 1 / 3);

    math::float3 result = math::float3(r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi));
    return result.normalize();
}

__host__ __device__ static math::float3 get_random_float3_in_disk()
{
    // note(rtarun9) : Find an actual formula / computation for this.
    math::float3 random_float3_in_disk = get_random_float3_in_sphere();
    random_float3_in_disk.z = 0.0f;

    return random_float3_in_disk.normalize();
}
__host__ __device__ static math::float3 get_random_unit_float3()
{
    // If you get random float3 in cube, note that the range of values
    // is [-1, -1, -1] to [1, 1, 1].
    // The dist between diagonal corner and other corners is not the same.
    // This can lead to potentially non - uniform distribution, which is what we want to avoid since lot of float3's
    // will be along the (longest) diagonal in the cube. For that reason we first get random float3 in sphere, and then
    // normalize it. https://github.com/RayTracing/raytracing.github.io/discussions/941
    return get_random_float3_in_sphere().normalize();
}