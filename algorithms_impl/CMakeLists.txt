cmake_minimum_required(VERSION 3.1)
project(algo_cpp_impl)

set(CMAKE_CXX_STANDARD 17)

add_library(algo_cpp_impl SHARED library.cpp)
add_executable(algo_cpp_impl_test test.cpp)
