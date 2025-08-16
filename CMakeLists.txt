cmake_minimum_required(VERSION 3.15)
project(covalent_base_identification LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found; Python modules will not be built")
endif()
find_package(Eigen3 QUIET)
find_package(FFTW3 QUIET)
find_package(Torch QUIET)
find_package(cnpy QUIET)
find_package(Iir QUIET)

add_subdirectory(src/cpp/identification)