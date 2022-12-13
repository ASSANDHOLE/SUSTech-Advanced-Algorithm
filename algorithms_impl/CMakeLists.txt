cmake_minimum_required(VERSION 3.1)
project(algo_cpp_impl)

set(CMAKE_CXX_STANDARD 20)

# show all warnings
if (MSVC)
    add_compile_options(/Wall)
else()
    add_compile_options(-Wall -Wextra -Wno-strict-aliasing)
endif()

# check if <USE_OMP> is set
if(USE_OMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        # set preprocessor macro
        add_definitions(-DOMP_ENABLED)
    else()
        message(STATUS "OpenMP not found")
    endif()
endif()

add_library(algo_cpp_impl SHARED library.cpp)
add_executable(algo_cpp_impl_test test.cpp)
