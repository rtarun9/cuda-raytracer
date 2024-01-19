#pragma once

#include "math/float3.hpp"
#include "math/ray.hpp"

#include <optional>

// Forward declaration for hit details.
struct hit_details_t;

namespace material
{
    // A abstraction for all materials.
    // All materials must override the scatter_ray function.
    class material_t
    {
    public:
        __device__ constexpr material_t(const math::float3& albedo) : albedo(albedo) {}

        // Function returns a ray_t if the ray is scattered (and not absorbed by the surface).
        __device__ virtual std::optional<math::ray_t> scatter_ray(const math::ray_t& ray, const hit_details_t &hit_details) const = 0;

    public:
        math::float3 albedo{};
    };
}
