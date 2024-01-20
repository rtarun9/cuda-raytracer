#pragma once

#include "types.hpp"
#include "math/float3.hpp"

#include <string_view>

// A simple image abstraction that provides method to create file / write to a output image file (currently, only png files are supported).
// The write_to_file function is to be called after the cuda kernel writes to the output frame buffer.
class image_t
{
public:
    constexpr image_t(const u32 width, const u32 height) : width(width), height(height) {}

    // Note : the output extension must be .png.
    void write_to_file(u8* const frame_buffer, const std::string_view output_file_path);

private:
    static constexpr u32 num_channels = 1;

public:
    u32 width{};
    u32 height{};
};