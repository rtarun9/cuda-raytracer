#include "renderer.hpp"

#include "utils.hpp"

#include "random_num_gen.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <vector_types.h>

__device__ math::float3 clamp_color_to_range_0_1(math::float3 color)
{
    math::float3 result = color;

    if (result.x >= 1.0f)
        result.x = 1.0f;
    if (result.x <= 0.0f)
        result.x = 0.0f;

    if (result.y >= 1.0f)
        result.y = 1.0f;
    if (result.y <= 0.0f)
        result.y = 0.0f;

    if (result.z >= 1.0f)
        result.z = 1.0f;
    if (result.z <= 0.0f)
        result.z = 0.0f;

    return result;
}

__device__ math::float3 get_background_color(const float ray_dir_y)
{
    const math::float3 white_color = math::float3(1.0f, 1.0f, 1.0f);
    const math::float3 sky_blue_color = math::float3(0.5f, 0.7f, 1.0f);

    return math::float3::lerp(white_color, sky_blue_color, (ray_dir_y + 1.0f) * 0.5f);
};

__global__ void raytracing_kernel(int sample_count, int max_depth, const math::float3 camera_center,
                                  const math::float3 upper_left_pixel_position, const math::float3 pixel_delta_u,
                                  const math::float3 pixel_delta_v, const math::float3 defocus_u,
                                  const math::float3 defocus_v, const scene::scene_t *scene, int image_width,
                                  int image_height, unsigned char *const frame_buffer)
{

    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row >= image_height || col >= image_width)
    {
        return;
    }

    math::float3 color{0.0f, 0.0f, 0.0f};

    for (int k = 0; k < sample_count; k++)
    {
        const math::float3 pixel_center =
            upper_left_pixel_position + pixel_delta_v * (float)(row) + pixel_delta_u * (float)(col);

        // Idea behind the math:
        // -0.5f + rand(0, 1) will be of range -0.5f, 0.5f.
        // pixel_delta is the distance between two pixels. When you multiply that by a factor in range 0.5f,
        // -0.5f, you get a position within the pixel 'grid'.
        const math::float3 pixel_sample = pixel_center + pixel_delta_u * get_random_float_in_range(-0.5f, 0.5f) +
                                          pixel_delta_v * get_random_float_in_range(-0.5f, 0.5f);

        // Compute defocus ray.
        const auto random_point_in_disc = get_random_float3_in_disk();
        const auto ray_origin = camera_center + defocus_u * random_point_in_disc.x + defocus_v * random_point_in_disc.y;
        const auto camera_to_pixel_ray = math::ray_t(ray_origin, (pixel_sample - ray_origin));

        math::ray_t ray = camera_to_pixel_ray;
        math::float3 per_sample_color = math::float3(1.0f, 1.0f, 1.0f);
        bool ray_absorbed = false;

        int i = 0;
        for (; i < max_depth; i++)
        {
            hit_details_t hit_record = scene->ray_hit(ray);

            if (hit_record.ray_param_t != -1.0f)
            {
                math::float3 albedo = scene->materials[hit_record.material_index].albedo;
                maybe_ray scatter_ray = scene->materials[hit_record.material_index].scatter_ray(ray, hit_record);

                if (scatter_ray.exists)
                {
                    per_sample_color = albedo * per_sample_color;
                    ray = scatter_ray.ray;
                    ray_absorbed = false;
                }
                else
                {
                    // If after contact with object the ray is not scattered, it is absorbed by the object.
                    per_sample_color = math::float3(0.0f, 0.0f, 0.0f);
                    ray_absorbed = true;
                    break;
                }
            }
            else
            {
                per_sample_color = per_sample_color * get_background_color(ray.direction.normalize().y);
                ray_absorbed = false;
                break;
            }
        };

        if (ray_absorbed || i == max_depth)
        {
            per_sample_color = math::float3(0.0f, 0.0f, 0.0f);
        }

        color += per_sample_color;
    };

    const auto inverse_sample_count = 1.0f / sample_count;
    color = color * inverse_sample_count;
    color = clamp_color_to_range_0_1(color);

    // Gamma correction.
    color = math::float3(sqrt(color.x), sqrt(color.y), sqrt(color.z));

    // Get flattend index of current pixel's contribution to framebuffer.
    int index = 3 * col + row * image_width * 3;
    frame_buffer[index] = color.x * 255;
    frame_buffer[index + 1] = color.y * 255;
    frame_buffer[index + 2] = color.z * 255;

    return;
}

__host__ u8 *renderer_t::render_scene(const scene::scene_t &scene, image_t &image) const
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
    // The pixel grid is inset from the viewport edges by half pixel to pixel distance. This means the pixel locations
    // are the actual locations of the pixel (i.e the center) and not the upper left corner of the square representing
    // the pixel. See https://raytracing.github.io/images/fig-1.04-pixel-grid.jpg for more details. This also makes the
    // viewport pixel grid evenly divided into w x h identical regions.s

    // Pixel delta : horizontal and vertical vectors between pixel location's.
    const auto pixel_delta_u = viewport_u / static_cast<float>(image.width);
    const auto pixel_delta_v = viewport_v / static_cast<float>(image.height);

    // Note that camera center (a point) + camera_front * focus_distance (a vector) equals a point. (i.e the
    // displacement of point by a vector yields a point).
    const auto viewport_upper_left =
        viewport_u * -0.5f + viewport_v * -0.5f + (camera_front * focus_distance + camera_center);
    const auto upper_left_pixel_position = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5f;

    // Prepare buffers for cuda kernel.
    u8 *host_frame_buffer = (unsigned char *)malloc(sizeof(u8) * image.width * image.height * 3);

    u8 *dev_frame_buffer = nullptr;
    utils::cuda_check(cudaMalloc(&dev_frame_buffer, (size_t)(sizeof(u8) * image.width * image.height * 3)));

    // Prepare kernel execution launch parameters.
    const dim3 threads_per_block = dim3(16, 16, 1);
    const dim3 blocks_per_grid = dim3((image.width + threads_per_block.x - 1) / threads_per_block.x,
                                      (image.height + threads_per_block.y - 1) / threads_per_block.y, 1u);

    scene::scene_t *dev_scene_ptr = nullptr;
    utils::cuda_check(cudaMalloc(&dev_scene_ptr, sizeof(scene::scene_t)));
    utils::cuda_check(
        cudaMemcpy(dev_scene_ptr, &scene, sizeof(scene::scene_t), cudaMemcpyKind::cudaMemcpyHostToDevice));

    raytracing_kernel<<<blocks_per_grid, threads_per_block>>>(
        sample_count, max_depth, camera_center, upper_left_pixel_position, pixel_delta_u, pixel_delta_v, defocus_u,
        defocus_v, dev_scene_ptr, image.width, image.height, dev_frame_buffer);

    cudaError_t last_error = cudaGetLastError();
    utils::cuda_check(last_error);

    std::cout << "Kernel execution complete" << std::endl;

    // Copy dev_frame_buffer into host frame_buffer.
    utils::cuda_check(cudaMemcpy(host_frame_buffer, dev_frame_buffer, 3 * sizeof(u8) * image.width * image.height,
                                 cudaMemcpyDeviceToHost));

    std::cout << "Data copied from device memory to host" << std::endl;

    utils::cuda_check(cudaFree(dev_frame_buffer));
    utils::cuda_check(cudaFree(dev_scene_ptr));

    return host_frame_buffer;
}
