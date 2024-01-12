#include "scene/scene.hpp"

#include "scene/sphere.hpp"
#include "math/ray.hpp"

#include "hit_details.hpp"

#include <climits>
#include <optional>

namespace scene
{
    void scene_t::add_sphere(const sphere_t& sphere)
    {
        spheres.emplace_back(sphere);
    }

    std::optional<hit_details_t> scene_t::ray_hit(const math::ray_t& ray) const
    {
        // The value of t is slightly greater than 0 because of shadow acne.
        // There are situations where due to floating point precision problems, we may have 
        // a case where the ray param at t is not the exact value, causing the intersection point to be 
        // beneath the surface rather than on it. This will cause the ray to continuously intersect the surface of the sphere,
        // causing continous intersections results in false - shadowing.
        // By setting min_t to be greater than 0, this problem can be resolved.
        float min_t = 0.001f;
        float max_t = std::numeric_limits<float>::infinity();

        hit_details_t ray_hit_details{};
        bool ray_hit_object_in_scene = false;

        for (const auto& sphere : spheres)
        {
            const auto t = sphere.hit_by_ray(ray, min_t, max_t);
            if (t.has_value())
            {
                // Fill hit_details struct.
                ray_hit_details.ray_param_t = *t;
                ray_hit_details.point_of_intersection = ray.at(*t);

                math::float3 normal = (ray_hit_details.point_of_intersection - sphere.center) / sphere.radius;

                // To find if the ray hit a back face or front face.
                // If the angle between normal and ray direction is greater than 90, then
                // the ray hit a front face.
                const auto normal_and_ray_direction_dot_product = math::float3::dot(ray.direction.normalize(), normal);
                if (normal_and_ray_direction_dot_product < 0.0f)
                {
                    ray_hit_details.back_face = false;
                    ray_hit_details.normal = normal;
                }
                else
                {
                    ray_hit_details.back_face = true;
                    ray_hit_details.normal = normal * -1.0f;
                }

                ray_hit_details.mat = &sphere.mat;

                max_t = *t;

                ray_hit_object_in_scene = true;
            }
        }

        if (!ray_hit_object_in_scene)
        {
            return std::nullopt;
        } 

        return ray_hit_details;
    }
}
