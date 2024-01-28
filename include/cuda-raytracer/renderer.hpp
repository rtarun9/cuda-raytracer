#pragma once

#include "math/float3.hpp"
#include "math/ray.hpp"

#include "image.hpp"
#include "scene/scene.hpp"
#include "utils.hpp"

// Abstraction for rendering a scene given camera position, scene, and frame_buffer (that will contain the final
// rendered output).

class renderer_t
{
  public:
    // Returns framebuffer (i.e the output of cuda kernel) in host-visible memory.
    __host__ void render_scene(const scene::scene_t &scene, u32 image_width, u32 image_height,
                               u32 *unified_frame_buffer, bool is_moving);

  public:
    // The vertical fov is used to determine viewport height.
    // The viewport width is computed in the render_scene function using images aspect ratio and viewport height.
    float vertical_fov{90.0f};

    // sample count is used for anti aliasing.
    u32 sample_count{100u};

    // Determines number of ray bounces.
    u32 max_depth{10u};

    // Camera settings.
    math::float3 camera_center{0.0f, 0.0f, 0.0f};
    math::float3 camera_look_at{0.0f, 0.0f, 1.0f};

    math::float3 camera_right{};
    math::float3 camera_up{};
    math::float3 camera_front{};

    // The distance from the camera to that plane where all objects appear in perfect focus.
    // As we move away from the focal distance, objects will appear to be linearly more blurrier.
    float focus_distance{};

    // The angle of tip of cone (base of cone : the circle centered about the camera center, and tip of cone is the
    // pixel location). Essentially, to simulate defocus blur, we take camera ray origin samples in a disk centered
    // around the actual camera center. If you consider this disk and the pixel sample point to constitute a cone, the
    // defocus angle is the angle at the tip of this cone. (In the vertical sense).
    float defocus_angle{};
};