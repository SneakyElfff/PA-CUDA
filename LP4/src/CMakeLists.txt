cmake_minimum_required(VERSION 3.29)
project(LP4 CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50")

add_executable(LP4 LP4.cu)

set_target_properties(LP4 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
