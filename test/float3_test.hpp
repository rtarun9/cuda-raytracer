#include <cassert>

#include "math/float3.hpp"

using namespace math;

void float3_test()
{
    // Magnitude test.
    {
        const math::float3 a(1.0f, 2.0f, 3.0f);

        const auto res = a.magnitude();
        assert(res == std::sqrt(1.0f * 1.0f + 2.0f * 2.0f + 3.0f * 3.0f) && "float3 magnitude failed!");
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
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);

        constexpr auto res = -a;
        static_assert(((res.r == -1.0f) && (res.g == -2.0f) && (res.b == -3.0f)) && "float3 negation failed!");
    }

    // Addition test.
    {
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);
        constexpr math::float3 b(-1.0f, -1.0f, -1.0f);

        constexpr auto res = a + b;
        static_assert(((res.r == 0.0f) && (res.g == 1.0f) && (res.b == 2.0f)) && "float3 addition failed!");
    }

    // Subtraction test.
    {
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);
        constexpr math::float3 b(-1.0f, -1.0f, -1.0f);

        constexpr auto res = a - b;
        static_assert(((res.r == 2.0f) && (res.g == 3.0f) && (res.b == 4.0f)) && "float3 subtraction failed!");
    }

    // float3 and scalar multiplication test.
    {
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);

        constexpr auto res = a * 2.0f;
        static_assert(((res.r == 2.0f) && (res.g == 4.0f) && (res.b == 6.0f)) && "float3 scalar multiplication failed!");
    }

    // float3 and scalar division test.
    {
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);

        constexpr auto res = a / 2.0f;
        static_assert(((res.r == 0.5f) && (res.g == 1.0f) && (res.b == 1.5f)) && "float3 scalar division failed!");
    }

    // Dot product test.
    {
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);
        constexpr math::float3 b(-1.0f, -1.0f, 5.0f);

        constexpr auto res = math::float3::dot(a, b);
        static_assert(res == 12.0f && "float3 dot product failed!");
    }

    // Lerp test.
    {
        constexpr math::float3 a(-1.0f, -2.0f, -3.0f);
        constexpr math::float3 b(1.0f, 2.0f, 3.0f);

        constexpr auto res_0 = math::float3::lerp(a, b, 0.0f);
        static_assert(res_0.r == -1.0f && res_0.g == -2.0f && res_0.b == -3.0f && "float3 lerp for t = 0.0f failed!");

        constexpr auto res_0_5 = math::float3::lerp(a, b, 0.5f);
        static_assert(res_0_5.r == 0.0f && res_0_5.g == 0.0f && res_0_5.b == 0.0f && "float3 lerp for t = 0.5f failed!");

        constexpr auto res_1 = math::float3::lerp(a, b, 1.0f);
        static_assert(res_1.r == 1.0f && res_1.g == 2.0f && res_1.b == 3.0f && "float3 lerp for t = 1.0f failed!");
    }
}