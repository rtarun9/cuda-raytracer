#pragma once

#include "material.hpp"

namespace material
{
    class lambertian_diffuse : public material_t
    {
    public:
        constexpr lambertian_diffuse(const math::float3& albedo) : material_t(albedo) {}

        std::optional<math::ray_t> scatter_ray(const math::ray_t& ray, const hit_details_t &hit_details) const;
    };
}