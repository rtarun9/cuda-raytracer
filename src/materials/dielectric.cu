#include "materials/dielectric.hpp"

#include "hit_details.hpp"
#include "utils.hpp"
#include "random_num_gen.hpp"

namespace material
{
    __device__ maybe_ray dielectric::scatter_ray(const math::ray_t& ray, const hit_details_t &hit_details) const
    {
        // The image being used for this derivation can be found in slide 13 of this ppt : 
        // https://web.cse.ohio-state.edu/~shen.94/681/Site/Slides_files/reflection_refraction.pdf.

        // The refracted ray is T, and the known values are N, V, theta i, theta t.
        // T is decomposed into sum of parallel and perpendicular components.
        // i.e, T = a + b
        // T = M sin(theta t) - N cos(theta t)
        // Now, M is the unknown in this equation.
        // See the above half (The half with V, N, theta i, etc.
        // Here, decomposing V into parallel and perpendicular components, we have:
        // V = N cos(theta i) + (-M)sin(theta i)
        // M = (N cos(theta i) - v) / sin(theta i)
        // Substituting in the equation involving T,
        // T = sin(theta t) (N cos (theta i) - v) / sin(theta i) - N cos(theta t)
        // Now, using snells law, sin (theta i) ei = sin (theta t) et.
        // We do not know sin (theta t), so substitute that with sin (theta i) ei / et.
        // T = (ei / et) (N cos (theta i) - v) - N * (sqrt(1 - sin(theta t)*2)
        // T = (ei / et) (N cos (theta i) - v) - N * sqrt(1  - sin(theta i) * sin(theta i) * e * e) 
        // T = (ei / et) (N cos (theta i) - v) - (N ) * sqrt(1.0f - sin(theta i) * sin(theta i) * e * e)
        // Here, ei / et is basically ior (e)

        float refraction_ratio = hit_details.back_face ? ior : 1.0f / ior;

        const auto& N = hit_details.normal;
        const auto& V = -ray.direction.normalize();

        const auto cos_theta_i = min(math::float3::dot(V, N), 1.0f);
        const auto sin_theta_i = sqrt(1 - cos_theta_i * cos_theta_i);

        // Before proceeding, note that the formula for computation of T requires a sqrt(1 - (sin(theta_i) * e) ^ 2).
        // The terms inside the sqrt cannot be negative. We can use this condition to determine when ray should refract (i.e when the sqrt computation is possible)
        // and when to reflect (when this computation is impossible).

        // Reflect if the below condition is true.
        // Also, we will reflect in some cases where the angle between V and N is large (i.e in such grazing angles
        // any surface will start to exhibit some mirror like reflection properties.
        // We can use the schlick approximation for this.
        const auto schlick_approximation = [&](const auto cos_theta, const auto ior)
        {
            const auto r0 = (1.0f - ior) / (1.0f + ior);
            const auto r0_square = r0 * r0;

            return r0_square + (1 - r0_square) *  std::pow((1.0f - cos_theta), 5);
        };

        // Perform reflection if either refraction is not possible, or if reflectance is above a random value.
        if (refraction_ratio * sin_theta_i > 1.0f || schlick_approximation(cos_theta_i, refraction_ratio) > get_random_float_in_range_0_1())
        {
           return maybe_ray{hit_details.point_of_intersection, (ray.direction - N * 2.0f * math::float3::dot(N, ray.direction)).normalize()};
        }

        // Refract the ray if refraction is possible. 
        const auto T = (N * cos_theta_i  - V) * refraction_ratio - N * std::sqrt(1.0f - sin_theta_i * sin_theta_i * refraction_ratio * refraction_ratio); 

        return maybe_ray(hit_details.point_of_intersection, T.normalize());   
    }
}