#include "float3.hpp"
#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <vector_types.h>

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

__global__ void raytrace(float2 inverse_window_coords, size_t render_target_width, uint8_t *render_target)
{
    // Orthogonal projection is used for now.
    size_t xcoord = threadIdx.x + blockIdx.x * blockDim.x;
    size_t ycoord = threadIdx.y + blockIdx.y * blockDim.y;

    float2 screen_space_coords = float2{xcoord * inverse_window_coords.x, ycoord * inverse_window_coords.y};
    float2 ndc = float2{screen_space_coords.x * 2.0f - 1.0f, screen_space_coords.y * 2.0f - 1.0f};

    math::float3 ray_origin = math::float3(ndc.x, ndc.y, 0.0f);
    math::float3 ray_direction = ray_origin + math::float3(0.0f, 0.0f, 1.0f);
    ray_direction = ray_direction.normalize();

    // For now, there is a fixed sphere.
    const math::float3 sphere_center = math::float3{0.0f, 0.0f, 5.0f};
    const float sphere_radius = 1.0f;

    // For a point P(x, y, z) to be on / inside / outside the sphere, we have to compare :
    // (P - sphere.center) . (P - sphere.center) and sphere.radius ^ 2              --- (i)
    // (P.x - sp.center.x) ^ 2 + (P.y - sp.center.y) ^ 2 + (P.z - sp.center.z) ^ 2 and sp.radius^2.
    // P is a point on the ray with origin org and direction dir. The unknown here is ray parameter t.
    // So, P is essentially a point on ray with parametric form P = ray.org + ray.dir * t.
    // Substituting in equation (i),
    // (org + dir * t - sp.center) . (org + dir * t - sp.center) <=> sp.radius^2
    // (dir * t + org - sp.center) . (dir * t + org - sp.center) <=> sp.radius^2
    // (dir . dir) * t ^ 2 + dir * t * 2 * (org - sp.center) + org . org + sp.center . sp.center - 2 * (org . sp.center)
    // t ^ 2 (dir . dir) + t (2 * dir . (org - sp.center)) + (org . org + sp.center . sp.center - 2 * (org. sp.center))
    // t ^ 2 (dir . dir) + t (2 * dir . (org - sp.center)) + (org - sp.center) . (org - sp.center) - sp.radius ^ <=> 0

    // If the determinant is >= 0, there exist a value of t that satisfies the equation, which means the ray does indeed
    // hit the sphere at one or 2 points.
    // determinant = B^2 - 4 A C
    // Here, B = 2 * dir * (org - sp.center), A = dir . dir, C = (org - sp.center) . (org - sp.center) - sp.radius ^ 2
    // Where equation is A t ^ 2 + B t + C = 0.

    const auto ray_origin_minus_center = ray_origin - sphere_center;

    const auto b = 2.0f * math::float3::dot(ray_direction, ray_origin_minus_center);
    const auto a = math::float3::dot(ray_direction, ray_direction);
    const auto c = math::float3::dot(ray_origin_minus_center, ray_origin_minus_center) - sphere_radius * sphere_radius;

    const auto determinant = b * b - 4 * a * c;
    if (determinant >= 0.0f)
    {
        uint32_t pixel_index = (xcoord + ycoord * render_target_width) * 3;
        render_target[pixel_index + 0] = 0xff;
        render_target[pixel_index + 1] = 0x0f;
        render_target[pixel_index + 2] = 0xf0;
    }
}

int main(int argc, char **argv)
{
    cudaDeviceProp device_prop = {};
    cudaGetDeviceProperties(&device_prop, 0);
    std::cout << "Cuda device name :: " << device_prop.name << '\n';

    // Setup SDL.
    constexpr uint32_t WINDOW_WIDTH = 1080;
    constexpr uint32_t WINDOW_HEIGHT = 720;

    // SDL initialization and window creation.
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("SDL_Init failed (%s)", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    SDL_Window *sdl_window =
        SDL_CreateWindow("cuda-raytracer", static_cast<int>(WINDOW_WIDTH), static_cast<int>(WINDOW_HEIGHT), 0);
    if (sdl_window == nullptr)
    {
        SDL_Log("SDL_CreateWindow failed (%s)", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    SDL_Renderer *sdl_renderer = SDL_CreateRenderer(sdl_window, nullptr);
    SDL_Texture *sdl_texture = SDL_CreateTexture(sdl_renderer, SDL_PixelFormat::SDL_PIXELFORMAT_RGB24,
                                                 SDL_TEXTUREACCESS_STREAMING, WINDOW_WIDTH, WINDOW_HEIGHT);

    uint8_t *dev_render_target = nullptr;
    cudaMalloc((void **)&dev_render_target, (size_t)(3 * WINDOW_HEIGHT * WINDOW_WIDTH));

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

        uint8_t *pixels = nullptr;
        int pitch = 3 * WINDOW_WIDTH;

        SDL_LockTexture(sdl_texture, nullptr, (void **)&pixels, &pitch);

        constexpr dim3 grid_dim = dim3(WINDOW_WIDTH / 32, WINDOW_HEIGHT / 32, 1);
        constexpr dim3 block_dim = dim3(32, 32, 1);

        raytrace<<<grid_dim, block_dim>>>(float2{1.0f / WINDOW_WIDTH, 1.0f / WINDOW_HEIGHT}, WINDOW_WIDTH,
                                          dev_render_target);
        cudaError_t kernel_exec_error = cudaGetLastError();
        if (kernel_exec_error != CUDA_SUCCESS)
        {
            printf("Cuda error: %d\n", kernel_exec_error);
        };

        cudaError_t memcpy_exec_error = cudaMemcpy(pixels, dev_render_target, 3 * WINDOW_WIDTH * WINDOW_HEIGHT,
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