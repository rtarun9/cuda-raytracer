#include "materials/lambertian_diffuse.hpp"

#include "hit_details.hpp"
#include "utils.hpp"

namespace material
{
    math::ray_t lambertian_diffuse::scatter_ray(const math::ray_t& ray, const hit_details_t &hit_details) const
    {
        // Direction is basically intersection point + normal + random float in sphere.
        // However, since scattered ray direction is this direction - intersection point, 
        // that term is not included explicitly in direction.
        math::float3 direction = (hit_details.normal + utils::get_random_float3_in_sphere().normalize()).normalize();

        // Check to see if the direction of scattered ray is close to 0 (i.e it is in the opposite direction of normal).
        constexpr float epsilon = 0.000001f;
        if (std::fabs(direction.r) <= epsilon &&  std::fabs(direction.g) <= epsilon && std::fabs(direction.b) <= epsilon)
        {
            direction = hit_details.normal;
        }

        return math::ray_t(hit_details.point_of_intersection, direction); 
    };
}