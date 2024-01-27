#pragma once

#include "math/float3.hpp"
#include "math/ray.hpp"

#include "image.hpp"
#include "scene/scene.hpp"
#include "utils.hpp"

// Abstraction for rendering a scene given camera position, scene, and image (that will contain the final rendered
// output).

class renderer_t
{
  public:
    // Returns framebuffer (i.e the output of cuda kernel) in host-visible memory.
    __host__ u8 *render_scene(const scene::scene_t &scene, image_t &image) const;

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

    // The distance from the camera to that plane where all objects appear in perfect focus.
    // As we move away from the focal distance, objects will appear to be linearly more blurrier.
    float focus_distance{};

    // The angle of tip of cone (base of cone : the circle centered about the camera center, and tip of cone is the
    // pixel location). Essentially, to simulate defocus blur, we take camera ray origin samples in a disk centered
    // around the actual camera center. If you consider this disk and the pixel sample point to constitute a cone, the
    // defocus angle is the angle at the tip of this cone. (In the vertical sense).
    float defocus_angle{};
};