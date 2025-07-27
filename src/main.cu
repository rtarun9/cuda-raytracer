#include "float3.hpp"
#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <iostream>
#include <vector_types.h>

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

__global__ void raytrace(float2 inverse_window_coords, size_t render_target_width, size_t render_target_height,
                         math::float3 camera_pos, uint8_t *render_target)
{
    // Orthogonal projection is used for now.
    size_t xcoord = threadIdx.x + blockIdx.x * blockDim.x;
    size_t ycoord = threadIdx.y + blockIdx.y * blockDim.y;

    if (xcoord >= render_target_width || ycoord >= render_target_height)
    {
        return;
    }

    // Clear RT
    uint32_t pixel_index = (xcoord + ycoord * render_target_width) * 3;
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
    render_target[pixel_index + 2] = uint8_t(ray_origin_visualize.z * 255.0f);
}

int main(int argc, char **argv)
{
    cudaDeviceProp device_prop = {};
    cudaGetDeviceProperties(&device_prop, 0);
    std::cout << "Cuda device name :: " << device_prop.name << '\n';

    // Setup SDL.
    constexpr uint32_t WINDOW_WIDTH = 1920;
    constexpr uint32_t WINDOW_HEIGHT = 1080;

    // SDL initialization and window creation.
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("SDL_Init failed (%s)", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    SDL_Window *sdl_window =
        SDL_CreateWindow("cuda-raytracer", static_cast<int>(WINDOW_WIDTH), static_cast<int>(WINDOW_HEIGHT),
                         SDL_WINDOW_BORDERLESS | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (sdl_window == nullptr)
    {
        SDL_Log("SDL_CreateWindow failed (%s)", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    int client_width = 0;
    int client_height = 0;

    SDL_Renderer *sdl_renderer = SDL_CreateRenderer(sdl_window, nullptr);
    SDL_GetRenderOutputSize(sdl_renderer, &client_width, &client_height);
    std::cout << "Client area width and height :: " << client_width << ", " << client_height << '\n';
    SDL_Texture *sdl_texture = SDL_CreateTexture(sdl_renderer, SDL_PixelFormat::SDL_PIXELFORMAT_RGB24,
                                                 SDL_TEXTUREACCESS_STREAMING, client_width, client_height);

    uint8_t *dev_render_target = nullptr;
    cudaMalloc((void **)&dev_render_target, (size_t)(3 * client_width * client_height));

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

        uint8_t *pixels = nullptr;
        int pitch = 3 * client_width;

        SDL_LockTexture(sdl_texture, nullptr, (void **)&pixels, &pitch);

        dim3 grid_dim = dim3((client_width + 31) / 32, (client_height + 31) / 32, 1);
        dim3 block_dim = dim3(32, 32, 1);

        raytrace<<<grid_dim, block_dim>>>(float2{1.0f / client_width, 1.0f / client_height}, client_width,
                                          client_height, camera_pos, dev_render_target);
        cudaError_t kernel_exec_error = cudaGetLastError();
        if (kernel_exec_error != CUDA_SUCCESS)
        {
            printf("Cuda error: %d\n", kernel_exec_error);
        };

        cudaError_t memcpy_exec_error = cudaMemcpy(pixels, dev_render_target, 3 * client_width * client_height,
                                                   cudaMemcpyKind::cudaMemcpyDeviceToHost);
        if (memcpy_exec_error != CUDA_SUCCESS)
        {
            printf("Cuda error: %d\n", memcpy_exec_error);
        };

        SDL_UnlockTexture(sdl_texture);

        SDL_RenderClear(sdl_renderer);
        SDL_RenderTexture(sdl_renderer, sdl_texture, nullptr, nullptr);
        SDL_RenderPresent(sdl_renderer);
    }

    return 0;
}