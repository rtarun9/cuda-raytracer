#include "types.hpp"
#include "image.hpp"
#include "math/float3.hpp"
#include "scene/scene.hpp"
#include "random_num_gen.hpp"
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
    constexpr u32 image_height = std::max(static_cast<u32>((float)image_width / aspect_ratio), 1u);

    image_t image(image_width, image_height);

    // Renderer config setup.
    renderer_t renderer{};
    renderer.max_depth = 1;
    renderer.sample_count = 1;
    renderer.vertical_fov = 90.0f;
    renderer.camera_center = math::float3(0.0f, 0.0f, -1.0f);
    renderer.camera_look_at = math::float3(0.0f, 0.0f, 1.0f);
    renderer.defocus_angle = 0.6f;
    renderer.focus_distance = (renderer.camera_look_at - renderer.camera_center).len();

    // Setup Scene and Materials.
 auto ground_material = material::lambertian_diffuse(math::float3(0.5f, 0.5f, 0.0f));

    auto world = scene::scene_t();
    material::lambertian_diffuse* ground_material_ptr = nullptr;
    utils::cuda_check(cudaMallocManaged(&ground_material_ptr, sizeof(material::lambertian_diffuse)));
    ground_material_ptr = &ground_material;
    world.add_material(ground_material_ptr);

    auto sphere = scene::sphere_t(math::float3(0.0f, 1.0f, 1.0f),0.5f,  (uint32_t)world.get_current_mat_index());
    world.add_sphere(sphere);

    #if 0
    for (int a = -11; a < -100; ++a)
    {
        for (int b = -11; b < -10; ++b)
        {
            const auto choose_mat = get_random_float_in_range_0_1();
            const auto center = math::float3(a + 0.9f * get_random_float_in_range_0_1(), 0.2f, b + 0.9f * get_random_float_in_range_0_1());

            if ((center - math::float3(4.0f, 0.2f, 0.0f)).len() > 0.9f)
            {
                if (choose_mat < 0.8f)
                {
                    // Diffuse.
                    world.add_material(new material::lambertian_diffuse(get_random_float3_in_range(0.0f, 1.0f)));
                    auto sphere = scene::sphere_t(center, 0.2f,  world.get_current_mat_index());

                    world.add_sphere(sphere);
                }
                else if (choose_mat < 0.95)
                {
                    // Metal.
                    world.add_material(new material::metal(get_random_float3_in_range(0.5f, 1.0f), get_random_float_in_range(0.0f, 0.5f)));
                    auto sphere = scene::sphere_t(center, 0.2f,  world.get_current_mat_index());
                    world.add_sphere(sphere);
                }
                else
                {
                    // Glass.
                    world.add_material(new material::dielectric(1.5f));
                    auto sphere = scene::sphere_t(center,  0.2f,  world.get_current_mat_index());
                    world.add_sphere(sphere);
                }
            }
        }
    }

    world.add_material(new material::dielectric(1.5f));
    auto sphere2 = scene::sphere_t(math::float3(0.0f, 1.0f, 0.0f),  1.0f,  world.get_current_mat_index());
    world.add_sphere(sphere);

    world.add_material(new material::metal(math::float3(0.7f, 0.6f, 0.5f), 0.0f));
    auto sphere3 = scene::sphere_t(math::float3(4.0f, 1.0f, 0.0f), 1.0f, world.get_current_mat_index());
    world.add_sphere(sphere3);

    world.add_material(new material::lambertian_diffuse(math::float3(0.4f, 0.2f, 0.1f)));
    auto sphere4 = scene::sphere_t(math::float3(-4.0f, 1.0f, 0.0f), 1.0f, world.get_current_mat_index());
    world.add_sphere(sphere4);

#endif

    // Begin render loop.
    u8* frame_buffer = renderer.render_scene(world, image);

    // Write rendered image to file.
    image.write_to_file(frame_buffer, "output_image.png");

    free(frame_buffer);

    return EXIT_SUCCESS;
}