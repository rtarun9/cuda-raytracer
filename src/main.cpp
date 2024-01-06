#include "types.hpp"
#include "image.hpp"
#include "math/float3.hpp"
#include "scene/scene.hpp"
#include "renderer.hpp"

int main()
{
    // Image setup.
    constexpr float aspect_ratio = 16.0f / 9.0f;
    constexpr u32 image_width = 512;
    constexpr u32 image_height = std::max(static_cast<u32>(image_width / aspect_ratio), 1u);

    image_t image(image_width, image_height);

    renderer_t renderer{};
    // Conventions used : Left handed coordinate system.
    renderer.camera_center = math::float3(0.0f, 0.0f, 0.0f);

    // Scene objects.
    constexpr scene::sphere_t sphere = scene::sphere_t{
        .center = math::float3(0.0f, 0.0f, 1.0f),
        .radius = 0.5f,
    };

    constexpr scene::sphere_t background_sphere = scene::sphere_t{
        .center = math::float3(0.0f, -100.5f, 1.0f),
        .radius = 100.0f,
    };

    scene::scene_t world{};
    world.add_sphere(sphere);
    world.add_sphere(background_sphere);

    renderer.render_scene(world, image);

    image.write_to_file("output_image.png");

    return EXIT_SUCCESS;
}   