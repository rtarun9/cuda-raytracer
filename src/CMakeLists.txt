set (SOURCE_FILES
    main.cpp
    image.cpp
    math/float3.cpp
)

add_executable(cuda-raytracer ${SOURCE_FILES})

target_link_libraries(cuda-raytracer PRIVATE extern)
target_include_directories(cuda-raytracer PRIVATE ${CMAKE_SOURCE_DIR}/include/cuda-raytracer)