#include "types.hpp"
#include "image.hpp"
#include "math/float3.hpp"
#include "scene/scene.hpp"
#include "renderer.hpp"
#include "materials/material.hpp"
#include "materials/lambertian_diffuse.hpp"
#include "materials/metal.hpp"
#include "materials/dielectric.hpp"

int main()
{
    // Image setup.
    constexpr float aspect_ratio = 16.0f / 9.0f;
    constexpr u32 image_width = 512;
    constexpr u32 image_height = std::max(static_cast<u32>(image_width / aspect_ratio), 1u);

    image_t image(image_width, image_height);

    // Renderer config setup.
    renderer_t renderer{};
    renderer.camera_center = math::float3(0.0f, 0.0f, 0.0f);
    renderer.max_depth = 100;
    renderer.sample_count = 100;

    // Materials.
    constexpr material::lambertian_diffuse ground_sphere_mat = material::lambertian_diffuse(math::float3(0.8f, 0.8f, 0.0f));
    constexpr material::lambertian_diffuse mat_center = material::lambertian_diffuse(math::float3(0.1f, 0.2f, 0.5));
    constexpr material::dielectric mat_left = material::dielectric(1.5f);
    constexpr material::metal mat_right = material::metal(math::float3(0.8f, 0.6f, 0.2f), 0.01f);
    
    // Scene objects.   
    scene::scene_t world{};
    world.add_sphere(scene::sphere_t{.center = math::float3(0.0f, -100.5f, 1.0f), .radius = 100.0f, .mat = ground_sphere_mat});
    world.add_sphere(scene::sphere_t{.center = math::float3(0.0f, 0.0f, 1.0f), .radius = 0.5f, .mat = mat_center});
    world.add_sphere(scene::sphere_t{.center = math::float3(-1.0f, 0.0f, 1.0f), .radius = 0.5f, .mat = mat_left});
    world.add_sphere(scene::sphere_t{.center = math::float3(-1.0f, 0.0f, 1.0f), .radius = -0.4f, .mat = mat_left});
    world.add_sphere(scene::sphere_t{.center = math::float3(1.0f, 0.0f, 1.0f), .radius = 0.5f, .mat = mat_right});

    // Begin render loop.
    renderer.render_scene(world, image);

    // Write rendered image to file.
    image.write_to_file("output_image.png");

    return EXIT_SUCCESS;
}   