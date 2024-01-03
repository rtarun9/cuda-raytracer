add_executable(cuda-raytracer main.cpp)

target_link_libraries(cuda-raytracer PRIVATE extern)
target_include_directories(cuda-raytracer PRIVATE ${CMAKE_SOURCE_DIR}/include/cuda-raytracer)