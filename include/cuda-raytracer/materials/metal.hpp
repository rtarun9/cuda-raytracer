#pragma once

#include "material.hpp"

namespace material
{
    class metal : public material_t
    {
    public:
        // The fuzziness factor is basically the sphere in which a random float3 will be chosen to deviate the reflected ray slightly.
        constexpr metal(const math::float3 &albedo, const float fuzziness_factor = 0) : material_t(albedo), fuzziness_factor(fuzziness_factor) {}

        std::optional<math::ray_t> scatter_ray(const math::ray_t &ray, const hit_details_t &hit_details) const;

    public:
        float fuzziness_factor{0.0f};
    };
}