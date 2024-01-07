#include "materials/metal.hpp"

#include "hit_details.hpp"
#include "utils.hpp"

namespace material
{
    math::ray_t metal::scatter_ray(const math::ray_t& ray, const hit_details_t &hit_details) const
    {
        // The direction is a reflection of the viewing / incoming ray direction (reflected around the normal).
        // That -1 is present because cos is negative (as normal and incoming ray direction v have angle between them
        // > 90 degree).
        math::float3 direction = ray.direction + hit_details.normal * -1.0f * 2.0f * math::float3::dot(hit_details.normal, ray.direction);

        return math::ray_t(hit_details.point_of_intersection, direction.normalize()); 
    };
}