#pragma once

#include <iostream>
#include <cmath>
#include <format>

namespace math
{
    // A simple float3 / vec3 class that will be used throughout the raytracer.
    // Naming conventions are very similar to HLSL.
    class float3
    {
    public:
        // Basic constructors.
        float3() {}
        float3(const float r, const float g, const float b) : r(r), g(g), b(b) {}
        float3(const float x) : r(x), g(x), b(x) {}

        // Mathematic operations
        const float magnitude() const
        {
            return std::sqrtf(r * r + g * g + b * b);
        }

        const float len() const
        {
            return magnitude();
        }

        const float3 normalize() const
        {
            const float mag = magnitude();
            return float3(r / mag, g / mag, b / mag);
        }

        // Overloaded mathematical functions.
        float3 &operator+=(const float3 &other)
        {
            r += other.r;
            g += other.g;
            b += other.g;

            return *this;
        }

        float3 &operator-=(const float3 &other)
        {
            r -= other.r;
            g -= other.g;
            b -= other.g;

            return *this;
        }

        float3 operator-() const
        {
            return float3(-1.0f * r, -1.0f * g, -1.0f * b);
        }

        // Debugging functions.
        friend std::ostream &operator<<(std::ostream &out, const float3 f)
        {
            out << std::format("({}, {}, {})", f.r, f.g, f.b);
        }

        static float dot(const float3& x, const float3& y)
        {
            return x.r * y.r + x.g * y.g + x.b * y.b;
        }

    public:
        float r{0.0f};
        float g{0.0f};
        float b{0.0f};
    };

    inline float3 operator+(const float3 &x, const float3 &y)
    {
        return float3(x.r + y.r, x.g + y.g, x.b + y.b);
    }
    inline float3 operator-(const float3 &x, const float3 &y)
    {
        return float3(x.r - y.r, x.g - y.g, x.b - y.b);
    }
}