cmake_minimum_required(VERSION 3.29)
project(LP1)

set(CMAKE_CXX_STANDARD 20)

# Установка флагов компиляции для оптимизации
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")  # O2
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")  # O3

add_executable(LP1 main.cpp)
