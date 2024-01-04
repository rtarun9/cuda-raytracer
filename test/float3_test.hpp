#include <cassert>

#include "math/float3.hpp"

void float3_test()
{
    // Magnitude test.
    {
        const math::float3 a(1.0f, 2.0f, 3.0f);

        const auto res = a.magnitude();
        assert(res == std::sqrtf(1.0f * 1.0f + 2.0f * 2.0f + 3.0f * 3.0f) && "float3 magnitude failed!");
    }

    // Len test.
    {
        const math::float3 a(1.0f, 2.0f, 3.0f);

        const auto res = a.len();
        assert(res == std::sqrtf(14.0f) && "float3 len failed!");
    }

    // normalize test.
    {
        const math::float3 a(1.0f, 2.0f, 3.0f);

        const auto res = a.normalize();
        const auto denom = std::sqrtf(14.0f);

        assert(((res.r == 1.0f / denom) && (res.g == 2.0f / denom) && (res.b == 3.0f / denom)) && "float3 normalize failed!");
    }

    // += test.
    {
        math::float3 a(1.0f, 2.0f, 3.0f);
        const math::float3 b(-1.0f, -1.0f, -1.0f);

        a += b;
        assert(((a.r == 0.0f) && (a.g == 1.0f) && (a.b == 2.0f)) && "float3 += failed!");
    }

    // -= test.
    {
        math::float3 a(1.0f, 2.0f, 3.0f);
        const math::float3 b(-1.0f, -1.0f, -1.0f);

        a -= b;
        assert(((a.r == 2.0f) && (a.g == 3.0f) && (a.b == 4.0f)) && "float3 -= failed!");
    }

    // negation test.
    {
        const math::float3 a(1.0f, 2.0f, 3.0f);

        const auto res = -a;
        assert(((res.r == -1.0f) && (res.g == -2.0f) && (res.b == -3.0f)) && "float3 negation failed!");
    }

    // Addition test.
    {
        const math::float3 a(1.0f, 2.0f, 3.0f);
        const math::float3 b(-1.0f, -1.0f, -1.0f);

        const auto res = a + b;
        assert(((res.r == 0.0f) && (res.g == 1.0f) && (res.b == 2.0f)) && "float3 addition failed!");
    }

    // Subtraction test.
    {
        const math::float3 a(1.0f, 2.0f, 3.0f);
        const math::float3 b(-1.0f, -1.0f, -1.0f);

        const auto res = a - b;
        assert(((res.r == 2.0f) && (res.g == 3.0f) && (res.b == 4.0f)) && "float3 subtraction failed!");
    }

    // float3 and scalar multiplication test.
    {
        const math::float3 a(1.0f, 2.0f, 3.0f);

        const auto res = a * 2.0f;
        assert(((res.r == 2.0f) && (res.g == 4.0f) && (res.b == 6.0f)) && "float3 scalar multiplication failed!");
    }

    // Dot product test.
    {
        const math::float3 a(1.0f, 2.0f, 3.0f);
        const math::float3 b(-1.0f, -1.0f, 5.0f);

        const auto res = math::float3::dot(a, b);
        assert(res == 12.0f && "float3 dot product failed!");
    }
}