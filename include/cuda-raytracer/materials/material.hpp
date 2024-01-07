#pragma once

#include "math/float3.hpp"
#include "math/ray.hpp"

struct hit_details_t;

namespace material
{
    // A abstraction for all materials.
    // All materials must override the scatter_ray function.
    class material_t
    {
    public:
        constexpr material_t(const math::float3& albedo) : albedo(albedo) {}

        virtual math::ray_t scatter_ray(const math::ray_t& ray, const hit_details_t &hit_details) const = 0;

    public:
        math::float3 albedo{};
    };
}
