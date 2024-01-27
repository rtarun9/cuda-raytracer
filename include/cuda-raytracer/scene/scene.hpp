#pragma once

#include "hit_details.hpp"
#include "materials/material.hpp"
#include "math/ray.hpp"
#include "sphere.hpp"

#include "types.hpp"

namespace scene
{
// Note that spheres and materials are placed in unified memory for convinience.
class scene_t
{
  public:
    __host__ scene_t(const u32 max_sphere_count = 30, const u32 max_material_count = 30);

    __host__ ~scene_t();

    __host__ uint32_t get_current_mat_index()
    {
        return num_materials - 1u;
    }

    __host__ void add_sphere(const sphere_t &sphere);

    // Adds material and returns current mat index.
    // Memory is handled by the scene object.
    __host__ uint32_t add_material(const material::material_t &mat);

    __device__ hit_details_t ray_hit(const math::ray_t &ray) const;

  public:
    uint32_t max_sphere_count{30u};
    uint32_t max_material_count{30u};

    sphere_t *spheres{};
    uint32_t num_spheres{};

    material::material_t *materials{};
    uint32_t num_materials{};
};
} // namespace scene