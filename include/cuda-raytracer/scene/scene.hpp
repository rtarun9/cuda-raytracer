#pragma once

#include <vector>
#include <optional>

#include "sphere.hpp"
#include "math/ray.hpp"

namespace scene
{
    // After a ray hits a object, the details at the point of intersection are stored in this struct.
    struct hit_details_t
    {
        math::float3 point_of_intersection;
        float ray_param_t{};

        math::float3 normal;
        bool back_face{false};
    };

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