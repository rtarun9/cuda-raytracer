add_executable(test float3_test.cpp)

target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR}/include/cuda-raytracer)