#include "renderer.hpp"

#include "utils.hpp"

#include <ranges>
#include <algorithm>
#include <functional>

void renderer_t::render_scene(const scene::scene_t &scene, image_t &image) const
{
    // Viewport setup.
    // The viewport is a rectangular region / grid in the 3D world that contains the image pixel grid.
    // The viewport in some terms is like the virtual window we use to look into the 3d world.
    // In rasterization it is the near plane of the viewing frustum.
    const float viewport_height = 2.0f * tanf(utils::degree_to_radians(vertical_fov / 2.0f)) * focus_distance;
    const float viewport_width = viewport_height * (static_cast<float>(image.width) / static_cast<float>(image.height));

    std::cout << "Viewport width and height : " << viewport_width << ", " << viewport_height << std::endl;

    // Setup of camera vectors.

    // Camera direction vectors to establish basis vectors.
    const math::float3 camera_front = (camera_look_at - camera_center).normalize();

    const math::float3 world_up = math::float3(0.0f, 1.0f, 0.0f);
    const math::float3 camera_right = math::float3::cross(world_up, camera_front).normalize();
    const math::float3 camera_up = math::float3::cross(camera_front, camera_right).normalize();

    std::cout << "camera_front, right, up : " << camera_front << " " << camera_right << " " << camera_up << std::endl;

    // Vectors along the viewport edge, u and v. Useful for getting the world space from image space coordinates.
    const math::float3 viewport_u = camera_right * viewport_width;
    const math::float3 viewport_v = camera_up * -1.0f * viewport_height;

    // Computation for focal radius and defocus u and v.
    const float defocus_radius = tan(utils::degree_to_radians(defocus_angle) / 2.0f) * focus_distance;
    std::cout << "Defocus radius :: " << defocus_radius << std::endl;

    // Camera defocus basis vectors (the basis vectors are scaled by defocus radius).
    const math::float3 defocus_u = camera_right * defocus_radius;
    const math::float3 defocus_v = camera_up * defocus_radius;

    // The image coordinates (row, col) are in image space, where origin is top left.
    // But the rays the camera shoots to the scene must be in world space.
    // The camera_center and viewport vectors will help in this coordinate system conversion.
    // The pixel grid is inset from the viewport edges by half pixel to pixel distance. This means the pixel locations are the actual
    // locations of the pixel (i.e the center) and not the upper left corner of the square representing the pixel.
    // See https://raytracing.github.io/images/fig-1.04-pixel-grid.jpg for more details. This also makes the viewport pixel grid
    // evenly divided into w x h identical regions.s

    // Pixel delta : horizontal and vertical vectors between pixel location's.
    const auto pixel_delta_u = viewport_u / static_cast<float>(image.width);
    const auto pixel_delta_v = viewport_v / static_cast<float>(image.height);

    // Note that camera center (a point) + camera_front * focus_distance (a vector) equals a point. (i.e the displacement of point by a vector yields a point).
    const auto viewport_upper_left = viewport_u * -0.5f + viewport_v * -0.5f + (camera_front * focus_distance + camera_center);
    const auto upper_left_pixel_position = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5f;

    for (const auto row : std::views::iota(0u, image.height))
    {
        std::cout << "Progress : " << 100.0f * static_cast<float>(row) / image.height << "%\r";
        for (const auto col : std::views::iota(0u, image.width))
        {
            // Using u, v to find the world space coordinate of viewport pixel.
            // pixel_delta_v and u after multiplication with col and row are of range : [0, viewport_v], [0, viewport_u]

            math::float3 color{0.0f, 0.0f, 0.0f};

            for (const auto k : std::views::iota(0u, sample_count))
            {
                const math::float3 pixel_center = upper_left_pixel_position + pixel_delta_v * static_cast<float>(row) + pixel_delta_u * static_cast<float>(col);

                // Idea behind the math:
                // -0.5f + rand(0, 1) will be of range -0.5f, 0.5f.
                // pixel_delta is the distance between two pixels. When you multiply that by a factor in range 0.5f, -0.5f, you get a position within the pixel 'grid'.
                const math::float3 pixel_sample = pixel_center + pixel_delta_u * utils::get_random_float_in_range(-0.5f, 0.5f) + pixel_delta_v * utils::get_random_float_in_range(-0.5f, 0.5f);

                // Compute defocus ray.
                const auto random_point_in_disc = utils::get_random_float3_in_disk();
                const auto ray_origin = camera_center + defocus_u * random_point_in_disc.x + defocus_v * random_point_in_disc.y;
                const auto camera_to_pixel_ray = math::ray_t(ray_origin, (pixel_sample - ray_origin));

                const auto get_background_color = [&](const float ray_dir_y) -> math::float3
                {
                    constexpr auto white_color = math::float3(1.0f, 1.0f, 1.0f);
                    constexpr auto sky_blue_color = math::float3(0.5f, 0.7f, 1.0f);

                    return math::float3::lerp(white_color, sky_blue_color, (ray_dir_y + 1.0f) * 0.5f);
                };

                const std::function<math::float3(math::ray_t, u32)> get_color = [&](const math::ray_t ray, const u32 depth) -> math::float3
                {
                    // If depth >= max_depth and in previous function call the ray did hit a objects, just assume that 
                    // the ray is absorbed (i.e not reflected) by the object. This ray will not have any color and just be black.
                    if (depth >= max_depth)
                    {
                        return math::float3(0.0f, 0.0f, 0.0f);
                    }

                    if (const auto hit_record = scene.ray_hit(ray); hit_record.has_value())
                    {
                        auto direction = scene.materials[hit_record->material_index]->scatter_ray(ray, *hit_record);
                        if (direction.has_value())
                        {
                            return get_color(*direction, depth + 1) * scene.materials[hit_record->material_index]->albedo;
                        }

                        // If after contact with object the ray is not scattered, it is absorbed by the object.
                        return math::float3(0.0f, 0.0f, 0.0f);
                    }

                    return get_background_color(ray.direction.normalize().y);
                };

                color += get_color(camera_to_pixel_ray, 0);
            };

            const auto inverse_sample_count = 1.0f / sample_count;
            color = math::float3(std::clamp(color.x * inverse_sample_count, 0.0f, 1.0f), std::clamp(color.y * inverse_sample_count, 0.0f, 1.0f), std::clamp(color.z * inverse_sample_count, 0.0f, 1.0f));
            image.add_normalized_float3_to_buffer(color);
        }
    }
}
