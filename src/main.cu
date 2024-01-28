#include "image.hpp"
#include "materials/material.hpp"
#include "math/float3.hpp"
#include "random_num_gen.hpp"
#include "renderer.hpp"
#include "scene/scene.hpp"
#include "types.hpp"

#include <chrono>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"

// Note : SDL sample code from : https://wiki.libsdl.org/SDL3/README/cmake
// Ref used (for how to render into SDL surface without gl interop) :
// https://github.com/fsan/cuda_on_sdl/blob/master/main.cpp

#define SDL_MAIN_HANDLED
#include <SDL.h>

int main(int argc, char **argv)
{
    // Image setup.
    constexpr float aspect_ratio = 16.0f / 9.0f;
    constexpr u32 image_width = 1200;
    constexpr u32 image_height = std::max(static_cast<u32>((float)image_width / aspect_ratio), 1u);

    // SDL initialization and window creation.
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
    {
        SDL_Log("SDL_Init failed (%s)", SDL_GetError());
        return -1;
    }

    SDL_Window *window =
        SDL_CreateWindow("cuda-raytracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                         static_cast<int>(image_width), static_cast<int>(image_height), SDL_WINDOW_SHOWN);
    if (window == nullptr)
    {
        SDL_Log("SDL_CreateWindow failed (%s)", SDL_GetError());
        return -1;
    }

    SDL_Surface *sdl_surface =
        SDL_CreateRGBSurface(0, image_width, image_height, 32, 0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF);

    if (sdl_surface == nullptr)
    {
        SDL_Log("Failed to create SDL surface %s", SDL_GetError());
        return -1;
    }

    SDL_Renderer *sdl_renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *sdl_texture = SDL_CreateTexture(sdl_renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING,
                                                 image_width, image_height);

    // Imgui setup.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForSDLRenderer(window, sdl_renderer);
    ImGui_ImplSDLRenderer2_Init(sdl_renderer);

    u32 *unified_frame_buffer = nullptr;
    utils::cuda_check(cudaMallocManaged(&unified_frame_buffer, sizeof(u32) * image_width * image_height));

    renderer_t renderer{};
    renderer.max_depth = 2;
    renderer.sample_count = 1;
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

    bool quit = false;
    float delta_time = 0.0f;
    auto previous_frame_time = std::chrono::high_resolution_clock::now();
    float movement_speed = 0.001f;

    int max_depth = 2;
    int sample_count = 1;

    while (!quit)
    {
        SDL_Event event;
        bool is_moving = false;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);

            if (event.type == SDL_QUIT)
            {
                quit = true;
            }

            const u8 *keyboard_state = SDL_GetKeyboardState(nullptr);
            if (keyboard_state[SDL_SCANCODE_ESCAPE])
            {
                quit = true;
            }

            if (keyboard_state[SDL_SCANCODE_W])
            {
                renderer.camera_center += renderer.camera_front * delta_time * movement_speed;
                is_moving = true;
            }

            if (keyboard_state[SDL_SCANCODE_S])
            {
                renderer.camera_center -= renderer.camera_front * delta_time * movement_speed;
                is_moving = true;
            }

            if (keyboard_state[SDL_SCANCODE_D])
            {
                renderer.camera_center += renderer.camera_right * delta_time * movement_speed;
                is_moving = true;
            }

            if (keyboard_state[SDL_SCANCODE_A])
            {
                renderer.camera_center -= renderer.camera_right * delta_time * movement_speed;
                is_moving = true;
            }
        }

        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ImGui::SliderInt("max_depth", &max_depth, 1, 50);
        renderer.max_depth = max_depth;

        ImGui::SliderInt("sample_count", &sample_count, 1, 50);
        renderer.sample_count = sample_count;

        if (is_moving)
        {
            renderer.sample_count = 1u;
            renderer.max_depth = 2u;
        }
        else
        {
            renderer.sample_count = sample_count;
            renderer.max_depth = max_depth;
        }

        SDL_LockSurface(sdl_surface);

        // Begin render loop.
        renderer.render_scene(world, image_width, image_height, unified_frame_buffer, is_moving);
        memcpy(sdl_surface->pixels, unified_frame_buffer, sizeof(u32) * image_width * image_height);

        SDL_UnlockSurface(sdl_surface);

        SDL_UpdateTexture(sdl_texture, nullptr, sdl_surface->pixels, sdl_surface->pitch);
        SDL_RenderClear(sdl_renderer);
        ImGui::Render();
        SDL_RenderCopy(sdl_renderer, sdl_texture, nullptr, nullptr);
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData());

        SDL_RenderPresent(sdl_renderer);

        delta_time = (float)std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() -
                                                                      previous_frame_time)
                         .count();
        previous_frame_time = std::chrono::high_resolution_clock::now();
    }

    cudaFree(unified_frame_buffer);

    return EXIT_SUCCESS;
}