#pragma once

#include "math/float3.hpp"

namespace material
{
    class material_t;
};

// After a ray hits a object, the details at the point of intersection are stored in this struct.
struct hit_details_t
{
    math::float3 point_of_intersection{};
    float ray_param_t{};

    math::float3 normal{};
    bool back_face{false};

    size_t material_index{};
};

