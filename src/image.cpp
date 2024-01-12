#include "image.hpp"

#include "stb/stb_image_write.h"

#include <filesystem>

void image_t::add_normalized_float3_to_buffer(const math::float3 &a)
{
    // The RGB color is being gamma corrected and converted to 0 -> 255 range here.
    // While gamma is 1/2.2f, we can use sqrt as a optimization for speed up.
    const auto rgb = std::vector{
        static_cast<u8>(255.0f * std::sqrt(a.x)),
        static_cast<u8>(255.0f * std::sqrt(a.y)),
        static_cast<u8>(255.0f * std::sqrt(a.z)),
    };

    buffer.insert(std::end(buffer), std::begin(rgb), std::end(rgb));
}

void image_t::write_to_file(const std::string_view output_file_path)
{
    const auto write_png = [&](const std::string_view file_path)
    {
        // Writing to file using stb.
        if (const int i = stbi_write_png(file_path.data(), width, height, num_channels, buffer.data(), width * num_channels); i)
        {
            std::cout << "Succesfully wrote to file : " << output_file_path;
        }
        else
        {
            std::cout << "Failed to write output to file : << output_file_path";
        }
    };

    if (std::filesystem::path(output_file_path).extension() != ".png")
    {
        std::cout << "When writing image to file, extension must be png!\n";
        std::cout << "Writing image to file default_output_file.png";

        write_png("default_output_file.png");

        return;
    }
    else
    {
        write_png(output_file_path);
    }
}
