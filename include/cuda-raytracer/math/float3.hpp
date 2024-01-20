#pragma once

#include <cmath>
#include <iostream>

namespace math
{
// A simple float3 / vec3 class that will be used throughout the raytracer.
// Naming conventions are very similar to HLSL.
class float3
{
  public:
    // Basic constructors.
    __host__ __device__ constexpr float3() noexcept
    {
    }
    __host__ __device__ constexpr float3(const float x, const float y, const float z) : x(x), y(y), z(z)
    {
    }
    __host__ __device__ constexpr float3(const float x) : x(x), y(x), z(x)
    {
    }

    // Mathematic operations

    // since std::sqrt is not constexpr, this function (and all others that use it)
    // are not marked as constexpr altogether.
    __host__ __device__ float magnitude() const
    {
#ifndef __CUDA_ARCH__

        return std::sqrt(x * x + y * y + z * z);
#else
        return sqrt(x * x + y * y + z * z);
#endif
    }

    __host__ __device__ float len() const
    {
        return magnitude();
    }

    __host__ __device__ float3 normalize() const
    {
        const float mag = magnitude();
        return float3(x / mag, y / mag, z / mag);
    }

    // Overloaded mathematical functions.
    __host__ __device__ float3 &operator+=(const float3 &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;

        return *this;
    }

    __host__ __device__ float3 &operator-=(const float3 &other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;

        return *this;
    }

    __host__ __device__ constexpr float3 operator+(const float3 &other) const
    {
        return float3(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ constexpr float3 operator-(const float3 &other) const
    {
        return float3(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ constexpr float3 operator-() const
    {
        return float3(-1.0f * x, -1.0f * y, -1.0f * z);
    }

    __host__ __device__ constexpr float3 operator*(const float t) const
    {
        return float3(x * t, y * t, z * t);
    }

    __host__ __device__ constexpr float3 operator*(const float3 &other) const
    {
        return float3(x * other.x, y * other.y, z * other.z);
    }

    __host__ __device__ constexpr float3 operator/(const float t) const
    {
        return float3(x / t, y / t, z / t);
    }

    __host__ __device__ static constexpr float dot(const float3 &a, const float3 &b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    // i    j   k
    // ax   ay  az
    // bx   by  bz
    __host__ __device__ static constexpr float3 cross(const float3 &a, const float3 &b)
    {
        return float3(a.y * b.z - a.z * b.y, -1.0f * (a.x * b.z - b.x * a.z), a.x * b.y - b.x * a.y);
    }

    // Returns (1.0f - t) * a + t * b;
    __host__ __device__ static constexpr float3 lerp(const float3 &a, const float3 &b, const float t)
    {
        return a * (1.0f - t) + b * t;
    }

    // Debugging functions.
    __host__ friend std::ostream &operator<<(std::ostream &out, const float3 f)
    {
        out <<  "(" << f.x << ", " << f.y << ", " << f.z << ")";
        return out;
    }

  public:
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
};
} // namespace math