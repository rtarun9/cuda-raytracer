#pragma once

#include "material.hpp"

namespace material
{
    class metal : public material_t
    {
    public:
        constexpr metal(const math::float3& albedo) : material_t(albedo) {}

        math::ray_t scatter_ray(const math::ray_t& ray, const hit_details_t &hit_details) const;
    };
}