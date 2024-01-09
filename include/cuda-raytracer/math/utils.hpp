#pragma once

#include <cmath>

namespace math::utils
{
    static inline float degree_to_radians(const float deg)
    {
        return (3.14159f / 180.0f) * deg;
    }
}