#include "materials/metal.hpp"

#include "hit_details.hpp"
#include "utils.hpp"
#include "random_num_gen.hpp"

namespace material
{
    __device__ maybe_ray metal::scatter_ray(const math::ray_t &ray, const hit_details_t &hit_details) const
    {
        // The direction is a reflection of the viewing / incoming ray direction (reflected around the normal).
        // That -1 is present because cos is negative (as normal and incoming ray direction v have angle between them
        // > 90 degree).
        // The formula for reflection of v around n is : -2 * (v.n) * n + v.
        math::float3 direction = ray.direction + hit_details.normal * -2.0f * math::float3::dot(hit_details.normal, ray.direction);
        direction = direction.normalize() + get_random_float3_in_sphere() * fuzziness_factor;
        direction = direction.normalize();

        // To introduce some fuzziness, the direction can be added with a random unit float3 and scaled with the fuzziness factor.
        // However do note that if the angle between the original direction and the normal (with fuziness factor taken into account
        // is > 90, then the ray will just be absorbed back by the surface, causing no further ray bounces into the scene,
        if (math::float3::dot(direction, hit_details.normal) > 0.0f)
        {
            return maybe_ray(hit_details.point_of_intersection, direction);
        }

        return maybe_ray();
    };
}