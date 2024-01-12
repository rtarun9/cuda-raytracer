#include "math/ray.hpp"

#include <cassert>

void ray_test()
{
    // at function test.
    {
        constexpr math::float3 origin{};
        constexpr math::float3 direction{1.0f, 5.0f, 3.0f};

        constexpr math::ray_t ray(origin, direction);
        
        constexpr const math::float3 at_5 = ray.at(5);
        static_assert(at_5.x== 5.0f && at_5.y == 25.0f && at_5.z == 15.0f && "ray_test at (5) failed");

        constexpr const math::float3 at_minus_5 = ray.at(-5);
        static_assert(at_minus_5.x == -5.0f && at_minus_5.y == -25.0f && at_minus_5.z == -15.0f && "ray_test at (-5) failed");
    }
}