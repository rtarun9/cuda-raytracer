#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <iostream>
#include <vector_types.h>

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "float3.hpp"
#include "sdl_wrapper.hpp"

__global__ void raytrace(float2 inverse_window_coords, uint2 render_target_size, math::float3 camera_pos,
                         uint8_t *render_target)
{
    // Orthogonal projection is used for now.
    size_t xcoord = threadIdx.x + blockIdx.x * blockDim.x;
    size_t ycoord = threadIdx.y + blockIdx.y * blockDim.y;

    if (xcoord >= render_target_size.x || ycoord >= render_target_size.y)
    {
        return;
    }

    // Clear RT
    uint32_t pixel_index = (xcoord + ycoord * render_target_size.x) * 3;
    render_target[pixel_index + 0] = 0x10;
    render_target[pixel_index + 1] = 0x10;
    render_target[pixel_index + 2] = 0x10;

    float2 screen_space_coords = float2{xcoord * inverse_window_coords.x, ycoord * inverse_window_coords.y};
    float2 ndc = float2{screen_space_coords.x * 2.0f - 1.0f, screen_space_coords.y * 2.0f - 1.0f};

    math::float3 ray_origin = math::float3(ndc.x, ndc.y, 0.0f) + camera_pos;
    math::float3 ray_direction = ray_origin + math::float3(0.0f, 0.0f, 1.0f);
    ray_direction = ray_direction.normalize();

    math::float3 ray_origin_visualize = ray_direction * 0.5f + 0.5f;
    render_target[pixel_index + 0] = uint8_t(ray_origin_visualize.x * 255.0f);
    render_target[pixel_index + 1] = uint8_t(ray_origin_visualize.y * 255.0f);
    // render_target[pixel_index + 2] = uint8_t(ray_origin_visualize.z * 255.0f);
}

int main(int argc, char **argv)
{
    cudaDeviceProp device_prop = {};
    cudaGetDeviceProperties(&device_prop, 0);
    std::cout << "Cuda device name :: " << device_prop.name << '\n';

    sdl_wrapper::render_context_t sdl_context(uint2{1920, 1080});
    const size_t render_surface_size_bytes = sdl_context.m_client_dim.x * sdl_context.m_client_dim.y * 3;
    const uint2 render_surface_size = sdl_context.m_client_dim;

    const float2 inverse_render_surface_size = {1.0f / render_surface_size.x, 1.0f / render_surface_size.y};

    uint8_t *dev_render_target = nullptr;
    cudaMalloc((void **)&dev_render_target, render_surface_size_bytes);

    math::float3 camera_pos = math::float3(0.0f, 0.0f, 0.0f);

    bool quit = false;
    while (!quit)
    {
        SDL_Event event = {};
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_EVENT_QUIT)
            {
                quit = true;
            }
        }

        const bool *keyboard_state = SDL_GetKeyboardState(nullptr);

        if (keyboard_state[SDL_SCANCODE_ESCAPE])
        {
            quit = true;
        }

        if (keyboard_state[SDL_SCANCODE_W])
        {
            camera_pos.z += 0.05f;
        }
        else if (keyboard_state[SDL_SCANCODE_S])
        {
            camera_pos.z -= 0.05f;
        }

        if (keyboard_state[SDL_SCANCODE_A])
        {
            camera_pos.x -= 0.05f;
        }
        else if (keyboard_state[SDL_SCANCODE_D])
        {
            camera_pos.x += 0.05f;
        }

        uint8_t *pixels = sdl_context.new_frame();

        dim3 grid_dim = dim3((render_surface_size.x + 31) / 32, (render_surface_size.y + 31) / 32, 1);
        dim3 block_dim = dim3(32, 32, 1);

        raytrace<<<grid_dim, block_dim>>>(inverse_render_surface_size, render_surface_size, camera_pos,
                                          dev_render_target);
        cudaError_t kernel_exec_error = cudaGetLastError();
        if (kernel_exec_error != CUDA_SUCCESS)
        {
            printf("Cuda error: %d\n", kernel_exec_error);
        };

        cudaError_t memcpy_exec_error =
            cudaMemcpy(pixels, dev_render_target, render_surface_size_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        if (memcpy_exec_error != CUDA_SUCCESS)
        {
            printf("Cuda error: %d\n", memcpy_exec_error);
        };

        sdl_context.render();
    }

    return 0;
}