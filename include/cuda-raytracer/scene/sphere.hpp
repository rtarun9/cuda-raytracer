#pragma once

#include "materials/material.hpp"
#include "math/float3.hpp"
#include "math/ray.hpp"

namespace scene
{
// The scene will have a list of materials, and the mat_index can be used to index into the material list.
struct sphere_t
{
    math::float3 center{};
    float radius{1.0f};
    size_t mat_index{};

    sphere_t(const math::float3 &center, float radius, size_t mat_index)
        : center(center), radius(radius), mat_index(mat_index)
    {
    }

    // A return value of -1.0f implies ray did not hit anything.
    __device__ float hit_by_ray(const math::ray_t &ray, const float min_t, const float max_t) const;
};
} // namespace scene