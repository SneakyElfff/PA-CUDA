cmake_minimum_required(VERSION 3.29)
project(LP2)

set(CMAKE_CXX_STANDARD 20)

# Установка флагов компиляции для оптимизации, AVX2 и OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -fopenmp")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -fopenmp")

add_executable(LP2 LP2.cpp)
