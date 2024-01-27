set (SOURCE_FILES
    main.cu
    image.cu
    renderer.cu
    scene/sphere.cu
    scene/scene.cu
    materials/material.cu
)

set (HEADER_FILES
    ../include/cuda-raytracer/materials/material.hpp
    ../include/cuda-raytracer/math/float3.hpp
    ../include/cuda-raytracer/math/ray.hpp
    ../include/cuda-raytracer/scene/scene.hpp
    ../include/cuda-raytracer/scene/sphere.hpp
    ../include/cuda-raytracer/hit_details.hpp
    ../include/cuda-raytracer/image.hpp
    ../include/cuda-raytracer/random_num_gen.hpp
    ../include/cuda-raytracer/renderer.hpp
    ../include/cuda-raytracer/types.hpp
    ../include/cuda-raytracer/utils.hpp

    )
find_package(CUDAToolkit)

add_executable(cuda-raytracer ${SOURCE_FILES} ${HEADER_FILES})

set_target_properties(cuda-raytracer PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda-raytracer PRIVATE extern CUDA::cudart CUDA::curand)

target_include_directories(cuda-raytracer PRIVATE "../include/cuda-raytracer")


