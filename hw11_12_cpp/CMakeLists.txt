cmake_minimum_required(VERSION 3.22)
project(hw11_12_cpp)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_PREFIX_PATH "../hw10_cpp/dependencies/or-tools")

find_package(ortools CONFIG REQUIRED)
if (NOT TARGET ortools::ortools)
    message(FATAL_ERROR "Could not find or-tools,
     following the instructions in
      ../hw10_cpp/depenencies/README.md to install")
endif ()

add_library(hw11_12_cpp SHARED main.cpp)
target_link_libraries(hw11_12_cpp ortools::ortools)

add_executable(hw11_12_cpp_test test.cpp)
target_link_libraries(hw11_12_cpp_test ortools::ortools)
