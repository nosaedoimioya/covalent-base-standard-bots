set(IDENTIFICATION_DEPS)
if(Torch_FOUND)
    list(APPEND IDENTIFICATION_DEPS "${TORCH_LIBRARIES}")
endif()
if(Eigen3_FOUND)
    list(APPEND IDENTIFICATION_DEPS Eigen3::Eigen)
endif()
if(Iir_FOUND)
    list(APPEND IDENTIFICATION_DEPS iir::iir)
endif()
# Try modern config first (some installs ship cnpyConfig.cmake)
find_package(cnpy CONFIG QUIET)

# If no imported target, fall back to finding header+lib directly
if(NOT TARGET cnpy::cnpy)
  find_path(CNPY_INCLUDE_DIR NAMES cnpy.h
            HINTS /usr/local/include /opt/homebrew/include
            PATH_SUFFIXES include)
  find_library(CNPY_LIBRARY NAMES cnpy
               HINTS /usr/local/lib /opt/homebrew/lib
               PATH_SUFFIXES lib)

  if(CNPY_INCLUDE_DIR AND CNPY_LIBRARY)
    # Create a compatible imported target so the rest of your code stays the same
    add_library(cnpy::cnpy UNKNOWN IMPORTED)
    set_target_properties(cnpy::cnpy PROPERTIES
      IMPORTED_LOCATION "${CNPY_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CNPY_INCLUDE_DIR}")
    set(cnpy_FOUND TRUE)
  endif()
endif()

# Add to your deps if available
if(TARGET cnpy::cnpy)
  list(APPEND IDENTIFICATION_DEPS cnpy::cnpy)
endif()

# Make sure we're in the same directory scope that uses FFTW3
find_package(FFTW3 REQUIRED)  # harmless if already found upstream

# Normalize an FFTW target or fall back to variables
set(FFTW_TARGET "")
if(TARGET FFTW3::fftw3)
  set(FFTW_TARGET FFTW3::fftw3)
elseif(TARGET FFTW3::FFTW3)              # some packages use this
  set(FFTW_TARGET FFTW3::FFTW3)
elseif(TARGET FFTW3::fftw3f)             # single-precision variant
  set(FFTW_TARGET FFTW3::fftw3f)
endif()

if(pybind11_FOUND)
    pybind11_add_module(MapFitter MODULE MapFitter.cpp)
    target_link_libraries(MapFitter PRIVATE ${IDENTIFICATION_DEPS})

    pybind11_add_module(MapGeneration MODULE MapGeneration.cpp)
    # Only if we fell back to variables (no imported target):
    if(NOT FFTW_TARGET)
        if(NOT IS_ABSOLUTE "${FFTW3_LIBRARIES}")
            # Try to resolve an absolute path (add more HINTS as needed)
            find_library(FFTW3_LIB NAMES ${FFTW3_LIBRARIES}
                        HINTS /opt/homebrew/lib /usr/local/lib /usr/lib)
            if(FFTW3_LIB)
            set(FFTW3_LIBRARIES "${FFTW3_LIB}")
            endif()
        endif()

        target_include_directories(MapGeneration PRIVATE ${FFTW3_INCLUDE_DIRS})
        target_link_libraries(MapGeneration PRIVATE ${IDENTIFICATION_DEPS} ${FFTW3_LIBRARIES})
    endif()

    pybind11_add_module(SineSweepReader MODULE SineSweepReader.cpp SineSweepReaderData.cpp)
    target_link_libraries(SineSweepReader PRIVATE ${IDENTIFICATION_DEPS})

    pybind11_add_module(FineTuneModelGen MODULE FineTuneModelGen.cpp)
    target_link_libraries(FineTuneModelGen PRIVATE ${IDENTIFICATION_DEPS})

    pybind11_add_module(ProcessCalibrationData MODULE ProcessCalibrationData.cpp)
    target_link_libraries(ProcessCalibrationData PRIVATE ${IDENTIFICATION_DEPS})
else()
    message(STATUS "pybind11 not found; identification Python modules will not be built")
endif()

install(
    FILES
        MapFitter.hpp
        MapGeneration.hpp
        SineSweepReader.hpp
        ProcessCalibrationData.hpp
        FineTuneModelGen.hpp
    DESTINATION include/identification
)