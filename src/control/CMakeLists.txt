# CMakeLists.txt for the control module

# Find required dependencies
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Set output directories for all targets
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Create control library
add_library(control_lib
    include/BaseShaper.cpp
)

# Set include directories
target_include_directories(control_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)

# Set compile options
target_compile_features(control_lib PRIVATE cxx_std_17)

# Create Python module
pybind11_add_module(base_shaper
    include/BaseShaperBindings.cpp
    include/BaseShaper.cpp
)

# Set include directories for Python module
target_include_directories(base_shaper
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)

# Set compile options for Python module
target_compile_features(base_shaper PRIVATE cxx_std_17)

# Create test executable
add_executable(test_base_shaper
    include/test_base_shaper.cpp
)

# Link test executable with control library
target_link_libraries(test_base_shaper
    PRIVATE
        control_lib
        ${EIGEN3_LIBRARIES}
)

# Set include directories for test
target_include_directories(test_base_shaper
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)

# Set compile options for test
target_compile_features(test_base_shaper PRIVATE cxx_std_17)

# Install rules
install(TARGETS control_lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS base_shaper
    LIBRARY DESTINATION lib
)

install(FILES include/BaseShaper.hpp
    DESTINATION src/control/include
)
