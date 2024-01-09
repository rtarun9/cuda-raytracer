#include "renderer.hpp"

#include "utils.hpp"

#include <ranges>
#include <algorithm>
#include <functional>

void renderer_t::render_scene(const scene::scene_t &scene, image_t &image) const
{
    // Setup of camera vectors.
    const float focal_length = (camera_look_at - camera_center).magnitude();

    // Camera direction vectors to establish basis vectors.
    const math::float3 camera_front = (camera_look_at - camera_center).normalize();

    const math::float3 world_up = math::float3(0.0f, 1.0f, 0.0f);
    const math::float3 camera_right = math::float3::cross(world_up,camera_front).normalize(); 
    const math::float3 camera_up = math::float3::cross(camera_front, camera_right).normalize();

    std::cout << "camera_front, right, up : " << camera_front << " " << camera_right << " " << camera_up << std::endl;

    // Viewport setup.
    // The viewport is a rectangular region / grid in the 3D world that contains the image pixel grid.
    // The viewport in some terms is like the virtual window we are viewing the 3D world in.
    // In rasterization it is the near plane of the viewing frustum.
    const float viewport_height = 2.0f * tanf(math::utils::degree_to_radians(vertical_fov / 2.0f)) * focal_length;
    const float viewport_width = viewport_height * (static_cast<float>(image.width) / static_cast<float>(image.height));

    std::cout << "Viewport width and height : " << viewport_width << ", " << viewport_height << std::endl;

    // Vectors along the viewport edge, u and v. Useful for getting the view space coordinate from image space coordinates.
    const math::float3 viewport_u = camera_right * viewport_width;
    const math::float3 viewport_v = camera_up * -1.0f * viewport_height;

    // The image coordinates (row, col) are in image space, where origin is top left.
    // But the rays the camera shoots to the scene must be in view space (where the camera is at center / origin).
    // The camera_center and viewport vectors will help in this coordinate system conversion.
    // The pixel grid is inset from the viewport edges by half pixel to pixel distance. This means the pixel locations are the actual
    // locations of the pixel (i.e the center) and not the upper left corner of the square representing the pixel.
    // See https://raytracing.github.io/images/fig-1.04-pixel-grid.jpg for more details. This also makes the viewport pixel grid
    // evenly divided into w x h identical regions.s

    // Pixel delta : horizontal and vertical vectors between pixel location's.
    const auto pixel_delta_u = viewport_u / static_cast<float>(image.width);
    const auto pixel_delta_v = viewport_v / static_cast<float>(image.height);

    const auto viewport_upper_left = math::float3(-viewport_u.r / 2.0f, -1.0f * viewport_v.g / 2.0f, focal_length) + (camera_look_at - camera_center);
    const auto upper_left_pixel_position = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5f;

    for (const auto row : std::views::iota(0u, image.height))
    {
        for (const auto col : std::views::iota(0u, image.width))
        {
            // Using u, v to find the view space coordinate of viewport pixel.
            // pixel_delta_v and u after multiplication with col and row are of range : [0, viewport_v], [0, viewport_u]

            math::float3 color{0.0f, 0.0f, 0.0f};

            for (const auto k : std::views::iota(0u, sample_count))
            {
                const math::float3 pixel_center = upper_left_pixel_position + pixel_delta_v * static_cast<float>(row) + pixel_delta_u * static_cast<float>(col);

                // Idea behind the math:
                // -0.5f + rand(0, 1) will be of range -0.5f, 0.5f.
                // pixel_delta is the distance between two pixels. When you multiply that by a factor in range 0.5f, -0.5f, you get a position within the pixel 'grid'.
                const math::float3 pixel_sample = pixel_center + pixel_delta_u * (-0.5f + utils::random_float_in_range_0_1()) + pixel_delta_v * (-0.5f + utils::random_float_in_range_0_1());

                const auto camera_to_pixel_ray = math::ray_t(camera_center, pixel_sample - camera_center);

                const auto get_background_color = [&](const float ray_dir_y) -> math::float3
                {
                    constexpr auto white_color = math::float3(1.0f, 1.0f, 1.0f);
                    constexpr auto sky_blue_color = math::float3(0.5f, 0.7f, 1.0f);

                    return math::float3::lerp(white_color, sky_blue_color, (ray_dir_y + 1.0f) * 0.5f);
                };

                const std::function<math::float3(math::ray_t, u32)> get_color = [&](const math::ray_t ray, const u32 depth) -> math::float3
                {
                    if (depth >= max_depth)
                    {
                        return math::float3(0.0f, 0.0f, 0.0f);
                    }

                    if (const auto hit_record = scene.ray_hit(ray); hit_record.has_value())
                    {
                        auto direction = hit_record->mat->scatter_ray(ray, *hit_record);
                        if (direction.has_value())
                        {
                            return get_color(*direction, depth + 1) * hit_record->mat->albedo;
                        }

                        return math::float3(0.0f, 0.0f, 0.0f);
                    }

                    return get_background_color(camera_to_pixel_ray.direction.normalize().g);
                };

                color += get_color(camera_to_pixel_ray, 0);
            };

            const auto inverse_sample_count = 1.0f / sample_count;
            color = math::float3(std::clamp(color.r * inverse_sample_count, 0.0f, 1.0f), std::clamp(color.g * inverse_sample_count, 0.0f, 1.0f), std::clamp(color.b * inverse_sample_count, 0.0f, 1.0f));
            image.add_normalized_float3_to_buffer(color);
        }
    }
}
