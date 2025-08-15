cmake_minimum_required(VERSION 3.15)
project(covalent_base_identification LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(pybind11 REQUIRED)
find_package(Eigen3 QUIET)
find_package(FFTW3 QUIET)
find_package(Torch QUIET)

add_subdirectory(cpp/identification)