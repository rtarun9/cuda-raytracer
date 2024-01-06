#pragma once

#include "math/float3.hpp"
#include "math/ray.hpp"

#include <optional>

namespace scene
{
    struct sphere_t
    {
        math::float3 center{};
        float radius{1.0f};
    
        const std::optional<float> hit_by_ray(const math::ray_t &ray) const;
    };
}