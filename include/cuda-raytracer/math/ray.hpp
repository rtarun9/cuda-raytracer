#pragma once

#include "float3.hpp"

namespace math
{
    // A ray is essentially of the form p(x) = a + t * b where a is origin, b is direction
    // and t is a parameter.
    // Use the p(x) function to get a point on the ray p.
    class ray_t
    {
    public:
        ray_t(const float3 origin, const float3 direction) : origin(origin), direction(direction) {}

        float3 at(const float t) const { return origin + direction * t; }

    public:
        float3 origin{0.0f, 0.0f, 0.0f};
        float3 direction{0.0f, 0.0f, 0.0f};
    };
}