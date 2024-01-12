#pragma once

#include "types.hpp"
#include "math/float3.hpp"

#include <vector>
#include <string>
#include <string_view>

// A simple image abstraction that provides method to create file / write to a output image file (currently, only png files are supported).
class image_t
{
public:
    constexpr image_t(const u32 width, const u32 height) : width(width), height(height) {}

    void add_normalized_float3_to_buffer(const math::float3 &x);

    // Note : the output extension must be .png.
    void write_to_file(const std::string_view output_file_path);

private:
    static constexpr u32 num_channels = 3;

public:
    u32 width{};
    u32 height{};

    std::vector<u8> buffer{};
};