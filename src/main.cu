#include "image.hpp"
#include "materials/material.hpp"
#include "math/float3.hpp"
#include "random_num_gen.hpp"
#include "renderer.hpp"
#include "scene/scene.hpp"
#include "types.hpp"

// Note : SDL sample code from : https://wiki.libsdl.org/SDL3/README/cmake
// Cuda-GL interop documentation :
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graphics-interoperability

#include "glad/include/glad/glad.h"

#define SDL_MAIN_HANDLED
#include <SDL.h>

int main(int argc, char **argv)
{
    // Image setup.
    constexpr float aspect_ratio = 16.0f / 9.0f;
    constexpr u32 image_width = 1200;
    constexpr u32 image_height = std::max(static_cast<u32>((float)image_width / aspect_ratio), 1u);

    image_t image(image_width, image_height);

    // SDL initialization and window creation.
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
    {
        SDL_Log("SDL_Init failed (%s)", SDL_GetError());
        return -1;
    }

    SDL_Window *window =
        SDL_CreateWindow("cuda-raytracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                         static_cast<int>(image_width), static_cast<int>(image_height), SDL_WINDOW_OPENGL);
    if (window == nullptr)
    {
        SDL_Log("SDL_CreateWindow failed (%s)", SDL_GetError());
        return -1;
    }

    // Create Opengl context (ref : https://gist.github.com/xeekworx/4c9a95c5eb67a1d3fc1fd35bacf84236)
    SDL_GLContext context = SDL_GL_CreateContext(window);
    if (context == nullptr)
    {
        SDL_Log("SDL_GL_CreateContext failed (%s)", SDL_GetError());
        return -1;
    }

    SDL_GL_MakeCurrent(window, context);

    // Initialize glad.
    if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress))
    {
        std::cout << "Failed to initialize glad.";
        return -1;
    }

    // Renderer config setup.
    renderer_t renderer{};
    renderer.max_depth = 40;
    renderer.sample_count = 35;
    renderer.vertical_fov = 20.0f;
    renderer.camera_center = math::float3(13.0, 2.0f, -3.0f);
    renderer.camera_look_at = math::float3(0.0f, 0.0f, 0.0f);
    renderer.defocus_angle = 0.6f;
    renderer.focus_distance = 10.0;

    // Setup Scene and Materials.
    auto world = scene::scene_t(500u, 500u);

    uint32_t idx = world.add_material(material::material_t().create_lambertain_diffuse(math::float3(0.5f, 0.5f, 0.5f)));
    auto sphere = scene::sphere_t(math::float3(0.0f, -1000.0f, 0.0f), 1000.0f, (uint32_t)idx);
    world.add_sphere(sphere);

    for (int a = -11; a < 11; ++a)
    {
        for (int b = -11; b < 11; ++b)
        {
            const auto choose_mat = get_random_float_in_range_0_1();
            const auto center = math::float3(a + 0.9f * get_random_float_in_range_0_1(), 0.2f,
                                             b + 0.9f * get_random_float_in_range_0_1());

            if ((center - math::float3(4.0f, 0.2f, 0.0f)).len() > 0.9f)
            {
                if (choose_mat < 0.8f)
                {
                    // Diffuse.
                    auto index = world.add_material(
                        material::material_t().create_lambertain_diffuse(get_random_float3_in_range(0.0f, 1.0f)));
                    auto sphere = scene::sphere_t(center, 0.2f, world.get_current_mat_index());

                    world.add_sphere(sphere);
                }
                else if (choose_mat < 0.95)
                {
                    // Metal.
                    world.add_material(material::material_t().create_metal(get_random_float3_in_range(0.5f, 1.0f),
                                                                           get_random_float_in_range(0.0f, 0.5f)));
                    auto sphere = scene::sphere_t(center, 0.2f, world.get_current_mat_index());
                    world.add_sphere(sphere);
                }
                else
                {
                    // Glass.
                    world.add_material(material::material_t().create_dielectric(1.5f));
                    auto sphere = scene::sphere_t(center, 0.2f, world.get_current_mat_index());
                    world.add_sphere(sphere);
                }
            }
        }
    }

    world.add_material(material::material_t().create_dielectric(1.5f));
    auto sphere2 = scene::sphere_t(math::float3(0.0f, 1.0f, 0.0f), 1.0f, world.get_current_mat_index());
    world.add_sphere(sphere2);

    world.add_material(material::material_t().create_metal(math::float3(0.7f, 0.6f, 0.5f), 0.0f));
    auto sphere3 = scene::sphere_t(math::float3(4.0f, 1.0f, 0.0f), 1.0f, world.get_current_mat_index());
    world.add_sphere(sphere3);

    world.add_material(material::material_t().create_lambertain_diffuse(math::float3(0.4f, 0.2f, 0.1f)));
    auto sphere4 = scene::sphere_t(math::float3(-4.0f, 1.0f, 0.0f), 1.0f, world.get_current_mat_index());
    world.add_sphere(sphere4);

    // Begin render loop.
    // u8 *frame_buffer = renderer.render_scene(world, image);

    glViewport(0, 0, image_width, image_height);

    bool quit = false;
    while (!quit)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                quit = true;
            }

            const u8 *keyboard_state = SDL_GetKeyboardState(nullptr);
            if (keyboard_state[SDL_SCANCODE_ESCAPE])
            {
                quit = true;
            }
        }

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        SDL_GL_SwapWindow(window);
    }

    // Write rendered image to file.
    // image.write_to_file(frame_buffer, "output_image.png");

    // cudaFree(frame_buffer);

    return EXIT_SUCCESS;
}