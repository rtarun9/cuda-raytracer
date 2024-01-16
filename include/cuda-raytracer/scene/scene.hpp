#pragma once

#include <vector>
#include <optional>

#include "sphere.hpp"
#include "math/ray.hpp"
#include "hit_details.hpp"
#include "materials/material.hpp"

namespace scene
{
    // A abstraction that contains a list of hittable objects.
    class scene_t
    {
    public:
        uint32_t get_current_mat_index() const { return static_cast<uint32_t>(materials.size() - 1); }

        void add_sphere(const sphere_t &sphere);

        std::optional<hit_details_t> ray_hit(const math::ray_t &ray) const;

    public:
        std::vector<sphere_t> spheres{};
        std::vector<std::shared_ptr<material::material_t>> materials;
    };
}