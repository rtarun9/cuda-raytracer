#include "image.hpp"

#include "stb/stb_image_write.h"

#include <filesystem>

void image_t::add_normalized_float3_to_buffer(const math::float3& x)
{
    buffer.emplace_back(static_cast<u8>(255.0f * x.r));
    buffer.emplace_back(static_cast<u8>(255.0f * x.g));
    buffer.emplace_back(static_cast<u8>(255.0f * x.b));
}

void image_t::write_to_file(const std::string_view output_file_path)
{
    if (std::filesystem::path(output_file_path).extension() != ".png")
    {
        std::cout << "When writing image to file, extension must be png!";
        return;
    }

    // Writing to file using stb.
    if (int i = stbi_write_png(output_file_path.data(), width, height, num_channels, buffer.data(), width * num_channels); i)
    {
        std::cout << "Succesfully wrote to file : " << output_file_path;
    }
}
