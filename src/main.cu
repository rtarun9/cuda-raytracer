#include <algorithm>
#include <cuda.h>
#include <iostream>

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

int main(int argc, char **argv)
{
    cudaDeviceProp device_prop = {};
    cudaGetDeviceProperties(&device_prop, 0);
    std::cout << "Cuda device name :: " << device_prop.name << '\n';

    // Setup SDL.
    constexpr float ASPECT_RATIO = 16.0f / 9.0f;
    constexpr uint32_t WINDOW_WIDTH = 1080;
    constexpr uint32_t WINDOW_HEIGHT = static_cast<uint32_t>((float)WINDOW_WIDTH / ASPECT_RATIO);

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
        int pitch = sizeof(float) * WINDOW_WIDTH;

        SDL_LockTexture(sdl_texture, nullptr, (void **)&pixels, &pitch);
        for (size_t i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++)
        {
            *pixels++ = 0xff;
        }
        SDL_UnlockTexture(sdl_texture);

        SDL_RenderClear(sdl_renderer);
        SDL_RenderTexture(sdl_renderer, sdl_texture, nullptr, nullptr);
        SDL_RenderPresent(sdl_renderer);
    }

    return 0;
}