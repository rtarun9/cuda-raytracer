set (SOURCE_FILES
    main.cu
    "image.cu"
    renderer.cu
    scene/sphere.cu
    scene/scene.cu
    materials/lambertian_diffuse.cu
    materials/metal.cu
    materials/dielectric.cu
)

find_package(CUDAToolkit)

add_executable(cuda-raytracer ${SOURCE_FILES})

set_target_properties(cuda-raytracer PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda-raytracer PRIVATE extern CUDA::cudart CUDA::curand)
target_include_directories(cuda-raytracer PRIVATE "../include/cuda-raytracer")

