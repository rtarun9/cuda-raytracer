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

        assert(((res.x == 1.0f / denom) && (res.y == 2.0f / denom) && (res.z == 3.0f / denom)) && "float3 normalize failed!");
    }

    // += test.
    {
        math::float3 a(1.0f, 2.0f, 3.0f);
        const math::float3 b(-1.0f, -1.0f, -1.0f);

        a += b;
        assert(((a.x == 0.0f) && (a.y == 1.0f) && (a.z == 2.0f)) && "float3 += failed!");
    }

    // -= test.
    {
        math::float3 a(1.0f, 2.0f, 3.0f);
        const math::float3 b(-1.0f, -1.0f, -1.0f);

        a -= b;
        assert(((a.x == 2.0f) && (a.y == 3.0f) && (a.z == 4.0f)) && "float3 -= failed!");
    }

    // negation test.
    {
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);

        constexpr auto res = -a;
        static_assert(((res.x == -1.0f) && (res.y == -2.0f) && (res.z == -3.0f)) && "float3 negation failed!");
    }

    // Addition test.
    {
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);
        constexpr math::float3 b(-1.0f, -1.0f, -1.0f);

        constexpr auto res = a + b;
        static_assert(((res.x == 0.0f) && (res.y == 1.0f) && (res.z == 2.0f)) && "float3 addition failed!");
    }

    // Subtraction test.
    {
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);
        constexpr math::float3 b(-1.0f, -1.0f, -1.0f);

        constexpr auto res = a - b;
        static_assert(((res.x == 2.0f) && (res.y == 3.0f) && (res.z == 4.0f)) && "float3 subtraction failed!");
    }

    // Element wise multiplciation test.
    {
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);
        constexpr math::float3 b(-1.0f, -1.0f, -1.0f);

        constexpr auto res = a * b;;
        static_assert(((res.x == -1.0f) && (res.y == -2.0f) && (res.z == -3.0f)) && "float3 element wise multiplication failed!");
    }


    // float3 and scalar multiplication test.
    {
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);

        constexpr auto res = a * 2.0f;
        static_assert(((res.x == 2.0f) && (res.y == 4.0f) && (res.z == 6.0f)) && "float3 scalar multiplication failed!");
    }

    // float3 and scalar division test.
    {
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);

        constexpr auto res = a / 2.0f;
        static_assert(((res.x == 0.5f) && (res.y == 1.0f) && (res.z == 1.5f)) && "float3 scalar division failed!");
    }

    // Dot product test.
    {
        constexpr math::float3 a(1.0f, 2.0f, 3.0f);
        constexpr math::float3 b(-1.0f, -1.0f, 5.0f);

        constexpr auto res = math::float3::dot(a, b);
        static_assert(res == 12.0f && "float3 dot product failed!");
    }

    // Cross product test.
    {
        constexpr math::float3 a(1.0f, 0.0f, 0.0f);
        constexpr math::float3 b(0.0f, 1.0f, 0.0f);

        constexpr auto res = math::float3::cross(a, b);
        static_assert(res.x == 0.0f&& res.y == 0.0f&& res.z == 1.0f&& "float3 dot product failed!");

        constexpr auto res2 = math::float3::cross(b, a);
        static_assert(res2.x == 0.0f&& res2.y == 0.0f&& res2.z == -1.0f&& "float3 dot product failed!");
    }

 
    // Lerp test.
    {
        constexpr math::float3 a(-1.0f, -2.0f, -3.0f);
        constexpr math::float3 b(1.0f, 2.0f, 3.0f);

        constexpr auto res_0 = math::float3::lerp(a, b, 0.0f);
        static_assert(res_0.x == -1.0f && res_0.y == -2.0f && res_0.z == -3.0f && "float3 lerp for t = 0.0f failed!");

        constexpr auto res_0_5 = math::float3::lerp(a, b, 0.5f);
        static_assert(res_0_5.x == 0.0f && res_0_5.y == 0.0f && res_0_5.z == 0.0f && "float3 lerp for t = 0.5f failed!");

        constexpr auto res_1 = math::float3::lerp(a, b, 1.0f);
        static_assert(res_1.x == 1.0f && res_1.y == 2.0f && res_1.z == 3.0f && "float3 lerp for t = 1.0f failed!");
    }
}