#include "materials/lambertian_diffuse.hpp"

#include "hit_details.hpp"
#include "utils.hpp"
#include "random_num_gen.hpp"

namespace material
{
   __device__ maybe_ray lambertian_diffuse::scatter_ray(const math::ray_t& ray, const hit_details_t &hit_details) const
    {
        // Direction is basically intersection point + normal + random float in sphere. Having the addition with normal will result in the direction
        // not being completely random, and rather be sort of close to the normal.
        // However, since scattered ray direction is this direction - intersection point, 
        // that term is not included explicitly in direction.
        math::float3 direction = (hit_details.normal + get_random_float3_in_sphere().normalize()).normalize();

        // Check to see if the direction of scattered ray is close to 0 (i.e it is in the opposite direction of normal).
        constexpr float epsilon = 0.000001f;
        if (std::fabs(direction.x) <= epsilon &&  std::fabs(direction.y) <= epsilon && std::fabs(direction.z) <= epsilon)
        {
            direction = hit_details.normal;
        }

        return maybe_ray(hit_details.point_of_intersection, direction); 
    };
}