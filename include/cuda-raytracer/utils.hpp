#pragma once

#include <random>
#include "math/float3.hpp"

namespace utils
{
    static inline float get_random_float_in_range_0_1()
    {
        static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        static std::mt19937 generator{};

        return distribution(generator);
    }

    static inline float get_random_float_in_range(const float min, const float max)
    {
        return min + (max - min) * get_random_float_in_range_0_1();
    }

    static constexpr float pi = 3.1415926535897932385f;

    static inline float degree_to_radians(const float deg)
    {
        return (pi / 180.0f) * deg;
    }

    static math::float3 get_random_float3_in_range(const float min, const float max)
    {
        return math::float3(get_random_float_in_range(min, max), get_random_float_in_range(min, max), get_random_float_in_range(min, max));
    }

    static math::float3 get_random_unit_float3()
    {
        return get_random_float3_in_range(0.0f, 1.0f);
    }

    static math::float3 get_float3_in_cube()
    {
        return math::float3(get_random_float_in_range(-1.0f, 1.0f),get_random_float_in_range(-1.0f, 1.0f), get_random_float_in_range(-1.0f, 1.0f));
    }

    static math::float3 get_random_float3_in_sphere()
    {
        // To get a random float3 in sphere, use the rejection method.
        // First get random float3 in cube (with w / h / d in range -1 to 1).
        // Then, if the length of the float3 is <= 1.0f, it is in the sphere.
        while (true)
        {
            const auto random_float3 = get_float3_in_cube();
            const auto mag = random_float3.magnitude();

            if (mag <= 1.0f)
            {
                return random_float3;
            }
        }
    }

    static math::float3 get_random_float3_in_disk()
    {
        while (true)
        {
            const auto random_float3 = math::float3(get_random_float_in_range(-1.0f, 1.0f), get_random_float_in_range(-1.0f, 1.0f), 0.0f);

            if (random_float3.len() <= 1.0f)
            {
                return random_float3;
            }
        }
    }
}