cmake_minimum_required(VERSION 3.22)
project(hw10_cpp)

set(CMAKE_CXX_STANDARD 20)

# add ortools path to CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH "dependencies/or-tools")

find_package(ortools CONFIG REQUIRED)

add_executable(hw10_cpp main.cpp)
target_link_libraries(hw10_cpp ortools::ortools)
