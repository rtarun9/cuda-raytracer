#include "scene/sphere.hpp"

#include "math/ray.hpp"

namespace scene
{
__device__ float sphere_t::hit_by_ray(const math::ray_t &ray, const float min_t, const float max_t) const
{
    // For a point P(x, y, z) to be on / inside / outside the sphere, we have to compare :
    // (P - sphere.center) . (P - sphere.center) and sphere.radius ^ 2              --- (i)
    // (P.x - sp.center.x) ^ 2 + (P.y - sp.center.y) ^ 2 + (P.z - sp.center.z) ^ 2 and sp.radius^2.
    // P is a point on the ray with origin org and direction dir. The unknown here is ray parameter t.
    // So, P is essentially a point on ray with parametric form P = ray.org + ray.dir * t.
    // Substituting in equation (i),
    // (org + dir * t - sp.center) . (org + dir * t - sp.center) <=> sp.radius^2
    // (dir * t + org - sp.center) . (dir * t + org - sp.center) <=> sp.radius^2
    // (dir . dir) * t ^ 2 + dir * t * 2 * (org - sp.center) + org . org + sp.center . sp.center - 2 * (org . sp.center)
    // t ^ 2 (dir . dir) + t (2 * dir . (org - sp.center)) + (org . org + sp.center . sp.center - 2 * (org. sp.center))
    // t ^ 2 (dir . dir) + t (2 * dir . (org - sp.center)) + (org - sp.center) . (org - sp.center) - sp.radius ^ <=> 0

    // If the determinant is >= 0, there exist a value of t that satisfies the equation, which means the ray does indeed
    // hit the sphere at one or 2 points.
    // determinant = B^2 - 4 A C
    // Here, B = 2 * dir * (org - sp.center), A = dir . dir, C = (org - sp.center) . (org - sp.center) - sp.radius ^ 2
    // Where equation is A t ^ 2 + B t + C = 0.

    // Optimization : See how B contains a 2 * .. term.
    // Say B = 2 X
    // Then, determinant = 4 X ^ 2 - 4 A C
    // So, the determinant becomes 4 * (X ^ 2 - A C) and the roots becomes
    // - X +- sqrt(determinant) / A

    const auto ray_origin_minus_center = ray.origin - center;

    const auto half_b = math::float3::dot(ray.direction, ray_origin_minus_center);
    const auto a = math::float3::dot(ray.direction, ray.direction);
    const auto c = math::float3::dot(ray_origin_minus_center, ray_origin_minus_center) - radius * radius;

    const auto determinant = half_b * half_b - a * c;
    if (determinant >= 0.0f)
    {
        const auto t1 = (-half_b - std::sqrt(determinant)) / (a);
        if (t1 >= min_t && t1 <= max_t)
        {
            return t1;
        }

        const auto t2 = (-half_b + std::sqrt(determinant)) / (a);
        if (t2 >= min_t && t2 <= max_t)
        {
            return t2;
        }
    }

    return -1.0f;
}
} // namespace scene
