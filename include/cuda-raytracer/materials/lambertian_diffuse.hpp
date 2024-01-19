#pragma once

#include "material.hpp"

namespace material
{
    class lambertian_diffuse : public material_t
    {
    public:
        __device__ constexpr lambertian_diffuse(const math::float3& albedo) : material_t(albedo) {}

        __device__ std::optional<math::ray_t> scatter_ray(const math::ray_t& ray, const hit_details_t &hit_details) const;
    };
}