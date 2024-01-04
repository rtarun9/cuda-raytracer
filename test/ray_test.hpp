#include "math/ray.hpp"

#include <cassert>

void ray_test()
{
    // at function test.
    {
        const math::float3 origin{};
        const math::float3 direction{1.0f, 5.0f, 3.0f};

        const math::ray_t ray(origin, direction);
        
        const math::float3 at_5 = ray.at(5);
        assert(at_5.r == 5.0f && at_5.g == 25.0f && at_5.b == 15.0f && "ray_test at (5) failed");

        const math::float3 at_minus_5 = ray.at(-5);
        assert(at_minus_5.r == -5.0f && at_minus_5.g == -25.0f && at_minus_5.b == -15.0f && "ray_test at (-5) failed");
    }
}