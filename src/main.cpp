#include <iostream>
#include <format>
#include <ranges>
#include <fstream>
#include <vector>

#include "types.hpp"
#include "image.hpp"
#include "math/float3.hpp"

int main()
{
    // Image setup.
    constexpr float aspect_ratio = 16.0f / 9.0f;
    constexpr u32 image_width = 5;
    constexpr u32 image_height = std::max(static_cast<u32>(image_width / aspect_ratio), 1u);

    image_t image(image_width, image_height);

    // Viewport setup.
    // The viewport is a rectangular region / grid which represents the part of the scene we are currently
    // viewing.
    // Note that aspect ratio is recomputed here, because this is the 'actual image aspect ratio' without integer rounding of
    // or checks to ensure image_height is atleast 1u. The aspect_ratio variable defined above is the ideal aspect ratio, but what is 
    // being computed for viewport is the 'actual' aspect ratio of the image.
    constexpr float viewport_width = 4.0f;
    constexpr float viewport_height = viewport_width / (static_cast<float>(image_width) / static_cast<float>(image_height));

    // Focal length : distance from the camera center to the viewport.
    constexpr float focal_length = 1.0f;

    // Conventions used : Left handed coordinate system.
    constexpr math::float3 camera_center = math::float3(0.0f, 0.0f, 0.0f);
    constexpr math::float3 viewport_left_extent = math::float3(-1.0f * viewport_width, 0.0f, 0.0f);
    constexpr math::float3 viewport_up_extent = math::float3(0.0f, viewport_height, 0.0f);

    // The render-loop coordinates (row, col) are in image space, where origin is top left. 
    // But the rays the camera shoots to the scene must be in view space (where the camera is at center / origin).
    // The camera_center, and viewport extents will help in this coordinate system conversion.

    for (const auto row : std::views::iota(0u, image_height))
    {
        for (const auto col : std::views::iota(0u, image_width))
        {
            // Normalization of row and col to the range of 0.0f to 1.0f.
            const auto u = static_cast<float>(row) / static_cast<float>(image_height - 1);
            const auto v = static_cast<float>(col) / static_cast<float>(image_width - 1);

            // Using u, v to find the view space coordinate of viewport pixel.
            const math::float3 pixel_position = viewport_left_extent * u + viewport_up_extent * (1.0f - v);
            std::cout << pixel_position << " ";

            auto rgb = math::float3();
            rgb.r = static_cast<float>(col) / static_cast<float>(image_width - 1);
            rgb.g = static_cast<float>(row) / static_cast<float>(image_height - 1);
            rgb.b = 0.0f;

            image.add_normalized_float3_to_buffer(rgb);
        }
        std::cout << "\n";
    }

    image.write_to_file("output.png");

    return EXIT_SUCCESS;
}