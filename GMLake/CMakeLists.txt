cmake_minimum_required(VERSION 3.16)
project(GMLake)

set(LIB_NAME gmlake)

if(DEBUG)
    add_compile_definitions(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    message(STATUS "Debug mode is enabled.")
else()
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Release mode is enabled.")
endif()

# use the pre C++ ABI instead of the C++11 ABI
add_definitions(-D _GLIBCXX_USE_CXX11_ABI=0)

# set CMAKE_CXX_STANDARD
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD   11)

# pytorch path
set(PYTORCH_DIR ${PYTORCH_DIR} CACHE PATH "Path to pytorch directory")

# CUDA runtime
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

add_subdirectory(src)