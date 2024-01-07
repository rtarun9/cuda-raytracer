set (SOURCE_FILES
    main.cpp
    image.cpp
    renderer.cpp
    scene/sphere.cpp
    scene/scene.cpp
    materials/lambertian_diffuse.cpp
    materials/metal.cpp
)

add_executable(cuda-raytracer ${SOURCE_FILES})

target_link_libraries(cuda-raytracer PRIVATE extern)
target_include_directories(cuda-raytracer PRIVATE ${CMAKE_SOURCE_DIR}/include/cuda-raytracer)

