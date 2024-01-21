#pragma once

#include "math/float3.hpp"
#include "math/ray.hpp"

// Forward declaration for hit details.
struct hit_details_t;

// Simplified version of std::optional<T>.
struct maybe_ray
{
    math::ray_t ray{};
    bool exists{false};

    __host__ __device__ maybe_ray()
    {
        exists = false;
    }

    __host__ __device__ maybe_ray(const math::float3 &origin, const math::float3 &direction)
        : ray({origin, direction}), exists(true)
    {
      exists = true;
    }
};

namespace material
{
// A abstraction for all materials.
// All materials must override the scatter_ray function.
class material_t
{
  public:
    __host__ constexpr material_t(const math::float3 &albedo) : albedo(albedo)
    {
    }

    // Function returns a ray_t if the ray is scattered (and not absorbed by the surface).
    __device__ virtual maybe_ray scatter_ray(const math::ray_t &ray, const hit_details_t &hit_details) const = 0;

  public:
    math::float3 albedo{};
};
} // namespace material
