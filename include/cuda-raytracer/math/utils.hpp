#pragma once

#include <random>

namespace utils
{
    float random_float_in_range_0_1()
    {
        static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        static std::mt19937 generator{};

        return distribution(generator);
    }
}