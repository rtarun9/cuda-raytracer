include(FetchContent)

add_library(extern STATIC stb/stb_image_write.h stb/stb_image_write.cpp glad/include/glad/glad.h glad/src/glad.cpp)

FetchContent_Declare(
    SDL2 
    GIT_REPOSITORY https://github.com/libsdl-org/SDL
    GIT_TAG prerelease-2.29.3
)

FetchContent_MakeAvailable(SDL2)

target_include_directories(extern PUBLIC "." ${sdl2_INCLUDE_DIRS})
target_link_libraries(extern PUBLIC SDL2::SDL2-static)