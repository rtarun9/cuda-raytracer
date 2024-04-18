include(FetchContent)

add_library(extern STATIC stb/stb_image_write.h stb/stb_image_write.cpp)

FetchContent_Declare(
    SDL2 
    GIT_REPOSITORY https://github.com/libsdl-org/SDL
    GIT_TAG prerelease-2.29.3
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui
    GIT_TAG 475453668d7da1662bf90266c5f0df0259705987
    GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(SDL2 imgui)

add_library(libimgui
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer2.cpp
)

target_include_directories(libimgui PUBLIC 
    ${imgui_SOURCE_DIR} 
    ${imgui_SOURCE_DIR}/backends
    ${SDL2_SOURCE_DIR}/include
)

target_include_directories(extern PUBLIC "." ${sdl2_INCLUDE_DIRS})
target_link_libraries(extern PUBLIC SDL2::SDL2-static libimgui)