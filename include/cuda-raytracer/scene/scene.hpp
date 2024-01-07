#pragma once

#include <vector>
#include <optional>

#include "sphere.hpp"
#include "math/ray.hpp"
#include "hit_details.hpp"

namespace scene
{
    // A abstraction that contains a list of hittable objects.
    class scene_t
    {
    public:
        void add_sphere(const sphere_t &sphere);

        std::optional<hit_details_t> ray_hit(const math::ray_t &ray) const;

    private:
        std::vector<sphere_t> spheres{};
    };
}