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

enum class material_type_t
{
    dielectric,
    metal,
    lambertian_diffuse,
};

namespace material
{
// A abstraction for all materials types.
// note(rtarun9) : Determine the possibility of having virtual functions / or some ECS so different material types can
// have thier own class abstraction.
class material_t
{
  public:
    __host__ __device__ material_t create_dielectric(const float index_of_refraction)
    {

        albedo = math::float3(1.0f, 1.0f, 1.0f);
        ior = index_of_refraction;

        mat_type = material_type_t::dielectric;

        return *this;
    }

    __host__ __device__ material_t create_lambertain_diffuse(const math::float3 &albedo)
    {
        this->albedo = albedo;

        mat_type = material_type_t::lambertian_diffuse;

        return *this;
    }

    __device__ __host__ material_t create_metal(const math::float3 &albedo, float fuzziness_factor)
    {
        this->albedo = albedo;
        this->fuzziness_factor = fuzziness_factor;

        mat_type = material_type_t::metal;

        return *this;
    }

    // Function ray_t if the ray is scattered (and not absorbed by the surface).
    __device__ maybe_ray scatter_ray(const math::ray_t &ray, const hit_details_t &hit_details) const;

  public:
    // Common variables for all material types.
    math::float3 albedo{};

    // Variables for dielectric materials.
    float ior{};

    // Variables for metal materials.
    float fuzziness_factor{0.0f};

    // To determine material type.
    material_type_t mat_type{};
};
} // namespace material
