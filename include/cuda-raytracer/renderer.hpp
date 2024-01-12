#pragma once

#include "math/ray.hpp"
#include "math/float3.hpp"

#include "image.hpp"
#include "scene/scene.hpp"
#include "utils.hpp"

// Abstraction for rendering a scene given camera position, scene, and image (that will contain the final rendered output).

class renderer_t
{
public:
    void render_scene(const scene::scene_t &scene, image_t &image) const;

public:
    float aspect_ratio{16.0f / 9.0f};
    float vertical_fov{90.0f};

    u32 sample_count{100u};
    u32 max_depth{10u};

    math::float3 camera_center{0.0f, 0.0f, 0.0f};
    math::float3 camera_look_at{0.0f, 0.0f, 1.0f};

    // The distance between the focal plane (where all objects are in perfect focus) and the camera.
    float focal_distance{};
    // The angle of tip of cone (base of cone : the circle centered about the camera center, and tip of cone is the pixel location).
    float defocus_angle{};
};