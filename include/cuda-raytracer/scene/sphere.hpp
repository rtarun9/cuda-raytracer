#pragma once

#include "math/float3.hpp"
#include "math/ray.hpp"
#include "materials/material.hpp"

#include <optional>

namespace scene
{
    struct sphere_t
    {
        math::float3 center{};
        float radius{1.0f};
        const material::material_t& mat;

        // Returns the ray parameter 't' if the ray hits the sphere and the value of t lies in the range
        // min_t and max_t. 
        const std::optional<float> hit_by_ray(const math::ray_t &ray, const float min_t, const float max_t) const;
    };
}