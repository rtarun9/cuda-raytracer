set (SOURCE_FILES
    main.cu
    image.cu
    renderer.cu
    scene/sphere.cu
    scene/scene.cu
    materials/lambertian_diffuse.cu
    materials/metal.cu
    materials/dielectric.cu
)

add_executable(cuda-raytracer ${SOURCE_FILES})

target_link_libraries(cuda-raytracer PRIVATE extern)
target_include_directories(cuda-raytracer PRIVATE ${CMAKE_SOURCE_DIR}/include/cuda-raytracer)

