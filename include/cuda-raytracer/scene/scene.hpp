#pragma once

#include <vector>
#include <optional>

#include "sphere.hpp"
#include "math/ray.hpp"
#include "hit_details.hpp"
#include "materials/material.hpp"

namespace scene
{
    // note(rtarun9) : TODO : Cleanup of allocated resources throughout the project.
    // A abstraction that contains a list of hittable objects.
    class scene_t
    {
    public:
        __host__ __device__ scene_t();

        uint32_t get_current_mat_index() { return num_materials - 1u;}

        void add_sphere(sphere_t &sphere);
        uint32_t add_material(material::material_t* mat);

        __device__ std::optional<hit_details_t> ray_hit(const math::ray_t &ray) const;

    public:
        uint32_t max_sphere_count{30u};
        uint32_t max_material_count{30u};

        sphere_t** spheres{};
        uint32_t num_spheres{};

        material::material_t** materials{};
        uint32_t num_materials{};
    };
}