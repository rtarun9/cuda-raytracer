#pragma once

#include "material.hpp"

namespace material
{
    // Dielectric materials can refract and reflect light rays.
    // In this implementation, whenever refraction is not possible, reflection will occur.
    class dielectric : public material_t
    {
    public:
        // Note that here, index_of_refraction is ir / it. Basically, ir is the outside medium refractive index (air in case ray is going from air to X).
        constexpr dielectric(const float index_of_refraction) : material_t(math::float3(1.0f, 1.0f, 1.0f)), ior(index_of_refraction) {}

        std::optional<math::ray_t> scatter_ray(const math::ray_t &ray, const hit_details_t &hit_details) const;

    public:
        float ior;
    };
}