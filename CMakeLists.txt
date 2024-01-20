cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)


project(
    cuda-raytracer
    VERSION 1.0
    LANGUAGES CXX CUDA
)

add_subdirectory(extern)
add_subdirectory(src)