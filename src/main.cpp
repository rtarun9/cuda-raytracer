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
    constexpr u32 image_width = 400;
    constexpr u32 image_height = std::max(static_cast<u32>(image_width / aspect_ratio), 1u);

    image_t image(image_width, image_height);

    // Renderer config setup.
    renderer_t renderer{};
    renderer.max_depth = 30;
    renderer.sample_count = 60;
    renderer.vertical_fov = 90.0f;  
    renderer.camera_center = math::float3(2.0f, 2.0f, -1.0f);
    renderer.camera_look_at = math::float3(0.0f, 0.0f, 1.0f);
    renderer.defocus_angle = 10.0f;
    renderer.focal_distance = 1.25f; 

    // Materials.
    const auto ground_material = material::lambertian_diffuse(math::float3(0.5f, 0.5f, 0.5f));

    auto world = scene::scene_t();
    world.add_sphere(scene::sphere_t{.center = math::float3(0.0f, -100.5f, 1.0f), .radius = 100.0f, .mat = ground_material});
    #if 0
    for (int a = -11; a < 11; ++a)
    {
        for (int b = -11; b < 11; ++b)
        {
            const auto choose_mat = utils::random_float_in_range_0_1();
            const auto center = math::float3(a + 0.9f * utils::random_float_in_range_0_1(), 0.2f, b + -0.9f * utils::random_float_in_range_0_1());

            if ((center - math::float3(4.0f, 0.2f, 0.0f)).len() > 0.9f) 
            {
                // Glass.
                const auto glass_material = material::dielectric(1.5f);
    
                if (choose_mat < 0.8f)
                {
                    // Diffuse.
                    const auto diffuse_mat = material::lambertian_diffuse(utils::get_random_float3_in_range_0_1());
                    world.add_sphere(scene::sphere_t{.center = center, .radius = 0.2f, .mat = diffuse_mat});
                }
                else if (choose_mat < 0.95)
                {
                    // Metal.
                    const auto metal_mat = material::metal(utils::get_random_float3_in_range_0_1(), utils::random_float_in_range_0_1() * 0.5f);
                    world.add_sphere(scene::sphere_t{.center = center, .radius=  0.2f, .mat = metal_mat});
                }
                else
                {
                    world.add_sphere(scene::sphere_t{.center = center, .radius = 0.2f, .mat = glass_material});
                }
            }
        }
    }
    #endif
    world.add_sphere(scene::sphere_t{.center = math::float3(1.0f, 0.0f, 1.0f), .radius = 0.5f, .mat = material::dielectric(1.5f)});
    world.add_sphere(scene::sphere_t{.center = renderer.camera_look_at, .radius = 0.5f, .mat = material::metal(math::float3(0.7f, 0.6f, 0.5f), 0.0f)});
    world.add_sphere(scene::sphere_t{.center = math::float3(-1.0f, 0.0f, 1.0f), .radius = 0.5f, .mat = material::lambertian_diffuse(math::float3(0.4f, 0.2f, 0.1f))});
    
    // Begin render loop.
    renderer.render_scene(world, image);

    // Write rendered image to file.
    image.write_to_file("output_image.png");

    return EXIT_SUCCESS;
}   