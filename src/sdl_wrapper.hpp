#pragma once

#include <vector_types.h>

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

namespace sdl_wrapper
{
    class render_context_t
    {
        public:
        render_context_t(uint2 window_dim): m_window_dim(window_dim)
        {
            // SDL initialization and window creation.
            if (!SDL_Init(SDL_INIT_VIDEO))
            {
                SDL_Log("SDL_Init failed (%s)", SDL_GetError());
            }

            m_window=
                SDL_CreateWindow("cuda-raytracer", static_cast<int>(m_window_dim.x), static_cast<int>(m_window_dim.y),
                                SDL_WINDOW_BORDERLESS | SDL_WINDOW_HIGH_PIXEL_DENSITY);
            if (m_window == nullptr)
            {
                SDL_Log("SDL_CreateWindow failed (%s)", SDL_GetError());
            }

            m_renderer = SDL_CreateRenderer(m_window, nullptr);

            int client_width = 0;
            int client_height = 0;
            SDL_GetRenderOutputSize(m_renderer, &client_width, &client_height);

            m_client_dim.x = client_width;
            m_client_dim.y = client_height;

            std::cout << "Client area width and height :: " << client_width << ", " << client_height << '\n';
            m_texture = SDL_CreateTexture(m_renderer, SDL_PixelFormat::SDL_PIXELFORMAT_RGB24,
                                                        SDL_TEXTUREACCESS_STREAMING, client_width, client_height);
        }

        // Returns pixels that can be rendered to.
        uint8_t* new_frame()
        {
            uint8_t *pixels = nullptr;
            int pitch = 3 * m_client_dim.x;

            SDL_LockTexture(m_texture, nullptr, (void **)&pixels, &pitch);
            return pixels;
        }

        void render()
        {
            SDL_UnlockTexture(m_texture);

            SDL_RenderClear(m_renderer);
            SDL_RenderTexture(m_renderer, m_texture, nullptr, nullptr);
            SDL_RenderPresent(m_renderer);
        }
        
        public:
        uint2 m_window_dim{};
        uint2 m_client_dim{};

        SDL_Window* m_window{};
        SDL_Texture* m_texture{};
        SDL_Renderer* m_renderer{};
};
}