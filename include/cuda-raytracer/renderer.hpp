#pragma once

#include "math/ray.hpp"
#include "math/float3.hpp"

#include "image.hpp"

#include "scene/scene.hpp"

// Abstraction for rendering a scene given camera position, scene, and image (that will contain the final rendered output).

class renderer_t
{
public:
    void render_scene(const scene::scene_t &scene, image_t &image) const;

public:
    float aspect_ratio{16.0f / 9.0f};
    math::float3 camera_center{0.0f, 0.0f, 0.0f};
    u32 sample_count{100u};
    u32 max_depth{10u};
};