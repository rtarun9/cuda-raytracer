#pragma once

#include "math/float3.hpp"
#include "math/ray.hpp"

#include <optional>

namespace scene
{
    // The scene will have a list of materials, and the mat_index can be used to index into the material list.
    struct sphere_t
    {
        math::float3 center{};
        float radius{1.0f};
        size_t mat_index{};

        // Returns the ray parameter 't' if the ray hits the sphere and the value of t lies in the range
        // min_t and max_t. 
        const std::optional<float> hit_by_ray(const math::ray_t &ray, const float min_t, const float max_t) const;
    };
}