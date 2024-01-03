#include <iostream>
#include <format>
#include <ranges>
#include <fstream>
#include <vector>

#include "types.hpp"
#include "math/float3.hpp"
#include "stb/stb_image_write.h"

std::vector<u8> normalized_float3_to_buffer(const math::float3 &x)
{
    return {static_cast<u8>(255.0f * x.r), static_cast<u8>(255.0f * x.g), static_cast<u8>(255.0f * x.b)};
}

int main()
{
    constexpr u32 width = 256 * 2;
    constexpr u32 height = 256;

    constexpr u32 max_color = 255;
    constexpr u32 num_channels = 3;

    std::vector<u8> buffer{};

    for (const auto row : std::views::iota(0u, height))
    {
        for (const auto col : std::views::iota(0u, width))
        {
            auto rgb = math::float3();
            rgb.r = static_cast<float>(col) / static_cast<float>(width - 1);
            rgb.g = static_cast<float>(row) / static_cast<float>(height - 1);
            rgb.b = 0.0f;

            const auto rgb_buffer = normalized_float3_to_buffer(rgb);
            buffer.insert(std::end(buffer), std::begin(rgb_buffer), std::end(rgb_buffer));
        }
    }

    // Writing to file using stb.
    if (int i = stbi_write_png("output.png", width, height, num_channels, buffer.data(), width * num_channels); i)
    {
        std::cout << "Succesfully wrote to file : output.png\n";
    }

    return EXIT_SUCCESS;
}