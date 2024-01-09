#pragma once

#include "material.hpp"

namespace material
{
    // Dielectric materials can refract and reflect light rays.
    // In this implementation, whenever refraction is not possible, reflection will occur.
    class dielectric : public material_t
    {
    public:
        // Here, the index of refraction of medium is to be specified. It will be assumed that if ray is hitting front face, the 
        // ray's ior is air (i.e 1). And, if the ray is exiting the surface, it is going into air (ior of 1).
        constexpr dielectric(const float index_of_refraction) : material_t(math::float3(1.0f, 1.0f, 1.0f)), ior(index_of_refraction) {}

        std::optional<math::ray_t> scatter_ray(const math::ray_t &ray, const hit_details_t &hit_details) const;

    public:
        float ior;
    };
}