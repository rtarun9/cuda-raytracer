#include <iostream>
#include <format>
#include <ranges>
#include <fstream>
#include <vector>

#include "types.hpp"
#include "image.hpp"
#include "math/float3.hpp"
#include "math/ray.hpp"

int main()
{
    // Image setup.
    constexpr float aspect_ratio = 16.0f / 9.0f;
    constexpr u32 image_width = 255;
    constexpr u32 image_height = std::max(static_cast<u32>(image_width / aspect_ratio), 1u);

    image_t image(image_width, image_height);

    // Viewport setup.
    // The viewport is a rectangular region / grid in the 3D world that contains the image pixel grid.
    // The viewport in some terms is like the virtual window we are viewing the 3D world in.
    // In rasterization it is the near plane of the viewing frustum.
    // https://www.gabrielgambetta.com/computer-graphics-from-scratch/02-basic-raytracing.html : We draw on the canvas / image
    // what ever we are seeing in the viewport.
    // Note that aspect ratio is recomputed here, because this is the 'actual image aspect ratio' without integer rounding of
    // or checks to ensure image_height is atleast 1u. The aspect_ratio variable defined above is the ideal aspect ratio, but what is 
    // being computed for viewport is the 'actual' aspect ratio of the image.
    constexpr float viewport_width = 4.0f;
    constexpr float viewport_height = viewport_width / (static_cast<float>(image_width) / static_cast<float>(image_height));

    // Focal length : distance from the camera center to the viewport.
    constexpr float focal_length = 1.0f;

    // Conventions used : Left handed coordinate system.
    constexpr math::float3 camera_center = math::float3(0.0f, 0.0f, 0.0f);

    // Vectors along the viewport edge, u and v.
    constexpr math::float3 viewport_u = math::float3(viewport_width, 0.0f, 0.0f);
    constexpr math::float3 viewport_v = math::float3(0.0f, -1.0f * viewport_height, 0.0f);

    // The image coordinates (row, col) are in image space, where origin is top left. 
    // But the rays the camera shoots to the scene must be in view space (where the camera is at center / origin).
    // The camera_center, and viewport vectors will help in this coordinate system conversion.
    // The pixel grid is inset from the viewport edges by half pixel to pixel distance. This means the pixel locations are the actual
    // locations of the pixel (i.e the center) and not the upper left corner of the square representing the pixel.
    // See https://raytracing.github.io/images/fig-1.04-pixel-grid.jpg for more details. This also makes the viewport pixel grid
    // evenly divided into w x h identical regions.s

    // Pixel delta : horizontal and vertical vectors between pixel location's.
    constexpr auto pixel_delta_u = viewport_u / image_width; 
    constexpr auto pixel_delta_v = viewport_v / image_height;

    constexpr auto viewport_upper_left = math::float3(-viewport_u.r / 2.0f, -1.0f * viewport_v.g / 2.0f, focal_length) - camera_center;
    constexpr auto upper_left_pixel_position = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5f;

    for (const auto row : std::views::iota(0u, image_height))
    {
        for (const auto col : std::views::iota(0u, image_width))
        {
            // Using u, v to find the view space coordinate of viewport pixel.
            // pixel_delta_v and u after multiplication with col and row are of range : [0, viewport_v], [0, viewport_u] 
            const math::float3 pixel_center = upper_left_pixel_position + pixel_delta_v * static_cast<float>(row) + pixel_delta_u * static_cast<float>(col); 
            const auto camera_to_pixel_ray = math::ray_t(camera_center, pixel_center - camera_center);

            const auto get_background_color = [&](const float ray_dir_y) -> math::float3
            {
                constexpr auto white_color =  math::float3(1.0f, 1.0f, 1.0f);
                constexpr auto teal_color = math::float3(0.1f, 0.9f, 0.7f);

                return math::float3::lerp(white_color, teal_color, (ray_dir_y + 1.0f) * 0.5f);
            };

            image.add_normalized_float3_to_buffer(get_background_color(camera_to_pixel_ray.direction.normalize().g));
        }
    }

    image.write_to_file("output.png");

    return EXIT_SUCCESS;
}