cmake_minimum_required(VERSION 3.30)

project(cuda-raytracer 
            VERSION 0.1 
            DESCRIPTION "Simple cuda raytracer"
            LANGUAGES CUDA CXX)

add_executable(cuda-raytracer "src/main.cu")

# External libs
include(FetchContent)

set(BUILD_SHARED_LIBS OFF)
FetchContent_Declare(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL
    GIT_TAG release-3.2.16
)

FetchContent_MakeAvailable(SDL3)

target_include_directories(cuda-raytracer PUBLIC ${SDL3_INCLUDE_DIRS})
target_link_libraries(cuda-raytracer PUBLIC SDL3::SDL3)
