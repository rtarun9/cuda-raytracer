#pragma once

#include <random>
#include "math/float3.hpp"

namespace utils
{
    float random_float_in_range_0_1()
    {
        static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        static std::mt19937 generator{};

        return distribution(generator);
    }

    math::float3 get_random_float3_in_range_0_1()
    {
        return math::float3(random_float_in_range_0_1(), random_float_in_range_0_1(), random_float_in_range_0_1());
    }

    math::float3 get_unit_float3_in_cube()
    {
        return math::float3(random_float_in_range_0_1() - 0.5f, random_float_in_range_0_1() - 0.5f, random_float_in_range_0_1() - 0.5f);
    }

    math::float3 get_random_float3_in_sphere()
    {
        // To get a random float3 in sphere, use the rejection method.
        // First get random float3 in cube (with w / h / d in range -1 to 1).
        // Then, if the length of the float3 is <= 1.0f, it is in the sphere.
        while (true)
        {
            const auto random_float3 = get_unit_float3_in_cube();
            const auto mag = random_float3.magnitude();

            if (mag <= 1.0f)
            {
                return random_float3;
            }
        } 
    }
}