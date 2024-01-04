add_executable(test main.cpp)

target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR}/include/cuda-raytracer)