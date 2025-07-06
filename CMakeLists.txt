cmake_minimum_required(VERSION 3.30)

project(cuda-raytracer 
            VERSION 0.1 
            DESCRIPTION "Simple cuda raytracer"
            LANGUAGES CUDA CXX)

add_executable(cuda-raytracer "src/main.cu")