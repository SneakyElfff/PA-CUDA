cmake_minimum_required(VERSION 3.29)
project(LP3 CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50")

add_executable(LP3 LP3.cu)

set_target_properties(LP3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
