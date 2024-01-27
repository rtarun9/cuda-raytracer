#include "image.hpp"

#include "stb/stb_image_write.h"

#include <filesystem>

void image_t::write_to_file(u8 *const frame_buffer, const std::string_view output_file_path)
{
    const auto write_png = [&](const std::string_view file_path) {
        // Writing to file using stb.
        if (const int i =
                stbi_write_png(file_path.data(), width, height, num_channels, frame_buffer, width * num_channels);
            i)
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
        std::cout << "Writing to png..." << std::endl;
        write_png(output_file_path);
    }
}
