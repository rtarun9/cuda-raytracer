#include "types.hpp"
#include "image.hpp"
#include "math/float3.hpp"
#include "scene/scene.hpp"
#include "renderer.hpp"
#include "materials/material.hpp"
#include "materials/lambertian_diffuse.hpp"
#include "materials/metal.hpp"
int main()
{
    // Image setup.
    constexpr float aspect_ratio = 16.0f / 9.0f;
    constexpr u32 image_width = 512;
    constexpr u32 image_height = std::max(static_cast<u32>(image_width / aspect_ratio), 1u);

    image_t image(image_width, image_height);

    renderer_t renderer{};
    renderer.camera_center = math::float3(0.0f, 0.0f, 0.0f);
    renderer.max_depth = 100;
    renderer.sample_count = 100;

    // Materials.
    constexpr material::lambertian_diffuse background_sphere_mat = material::lambertian_diffuse(math::float3(0.2f, 0.2f, 0.2f));
    constexpr material::lambertian_diffuse foreground_sphere_mat = material::lambertian_diffuse(math::float3(0.7f, 0.3f, 0.3f));
    constexpr material::metal foreground_sphere_mat_2 = material::metal(math::float3(0.8f, 0.8f, 0.8f));
    
    // Scene objects.
    const scene::sphere_t sphere = scene::sphere_t{
        .center = math::float3(0.0f, 0.0f, 1.0f),
        .radius = 0.5f,
        .mat = foreground_sphere_mat,
    };

    const scene::sphere_t sphere_2 = scene::sphere_t{
        .center = math::float3(-1.0f, 0.0f, 1.0f),
        .radius = 0.6f,
        .mat = foreground_sphere_mat_2,
    };

    const scene::sphere_t background_sphere = scene::sphere_t{
        .center = math::float3(0.0f, -100.5f, 1.0f),
        .radius = 100.0f,
        .mat = background_sphere_mat,
    };

    scene::scene_t world{};
    world.add_sphere(sphere);
    world.add_sphere(background_sphere);
    world.add_sphere(sphere_2);

    renderer.render_scene(world, image);

    image.write_to_file("output_image.png");

    return EXIT_SUCCESS;
}   