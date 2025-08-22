set(IDENTIFICATION_DEPS)
set(PYTHON_DEPS)
if(Torch_FOUND)
    list(APPEND IDENTIFICATION_DEPS "${TORCH_LIBRARIES}")
endif()
if(Eigen3_FOUND)
    list(APPEND IDENTIFICATION_DEPS Eigen3::Eigen)
endif()
# iir (support both naming conventions)
if(TARGET Iir::Iir)
  list(APPEND IDENTIFICATION_DEPS Iir::Iir)
elseif(TARGET iir::iir)
  list(APPEND IDENTIFICATION_DEPS iir::iir)
endif()

if(TARGET Python::Python)
  list(APPEND PYTHON_DEPS Python::Python)
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

add_library(ident_core STATIC CalibrationCore.cpp)

# --- loader libs ----------------------------------------------------------
add_library(LegacyMapLoader STATIC LegacyMapLoader.cpp)
set_target_properties(LegacyMapLoader PROPERTIES POSITION_INDEPENDENT_CODE ON)


add_library(NPZMapLoader STATIC NPZMapLoader.cpp)
target_link_libraries(NPZMapLoader PRIVATE ${IDENTIFICATION_DEPS})
target_include_directories(NPZMapLoader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(mapfitter_core STATIC MapFitterCore.cpp)
set_target_properties(mapfitter_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(sinesweep_core STATIC SineSweepReaderData.cpp)
set_target_properties(sinesweep_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(pybind11_FOUND)
    target_link_libraries(mapfitter_core
        PUBLIC
            pybind11::headers     # headers only; no Python linkage needed here
            ${IDENTIFICATION_DEPS} # Torch, Eigen, Iir, etc.
        )
    target_include_directories(mapfitter_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(ident_core PRIVATE pybind11::headers ${IDENTIFICATION_DEPS} ${PYTHON_DEPS})

    target_link_libraries(sinesweep_core
        PUBLIC
            pybind11::headers     # headers only; no Python linkage needed here
            ${IDENTIFICATION_DEPS}  # Eigen, Iir/IIR, Torch if needed
            ${PYTHON_DEPS}          # Python::Python if you call into Python in the core
        )
    target_include_directories(sinesweep_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(LegacyMapLoader PRIVATE pybind11::headers ${IDENTIFICATION_DEPS} ${PYTHON_DEPS})
    target_include_directories(LegacyMapLoader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    pybind11_add_module(MapFitter MODULE MapFitterBindings.cpp)
    target_link_libraries(MapFitter PRIVATE mapfitter_core ${IDENTIFICATION_DEPS})

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
        target_include_directories(ident_core PRIVATE ${FFTW3_INCLUDE_DIRS})
        target_link_libraries(ident_core PRIVATE ${FFTW3_LIBRARIES})

        target_include_directories(MapGeneration PRIVATE ${FFTW3_INCLUDE_DIRS})
        target_link_libraries(MapGeneration PRIVATE ${FFTW3_LIBRARIES})
    else()
        target_link_libraries(ident_core PRIVATE ${FFTW_TARGET})
        target_link_libraries(MapGeneration PRIVATE ${FFTW_TARGET})
    endif()
    
    target_link_libraries(MapGeneration PRIVATE ident_core ${IDENTIFICATION_DEPS})
    

    pybind11_add_module(SineSweepReader MODULE SineSweepReader.cpp)
    target_link_libraries(SineSweepReader 
        PRIVATE 
            ident_core
            sinesweep_core 
            ${IDENTIFICATION_DEPS}
        )

    pybind11_add_module(FineTuneModelGen MODULE FineTuneModelGen.cpp)
    target_link_libraries(FineTuneModelGen 
        PRIVATE 
            LegacyMapLoader
            NPZMapLoader
            ${IDENTIFICATION_DEPS})

    pybind11_add_module(ProcessCalibrationData MODULE ProcessCalibrationData.cpp)
    target_link_libraries(ProcessCalibrationData 
        PRIVATE 
            ident_core
            sinesweep_core 
            mapfitter_core
            LegacyMapLoader
            NPZMapLoader
            ${IDENTIFICATION_DEPS})

else()
    message(STATUS "pybind11 not found; identification Python modules will not be built")
endif()

foreach(tgt IN ITEMS MapGeneration MapFitter SineSweepReader FineTuneModelGen ProcessCalibrationData)
  if(TARGET ${tgt})
    target_link_options(${tgt} PRIVATE "-Wl,-undefined,error")
  endif()
endforeach()

install(
    FILES
        MapFitter.hpp
        MapGeneration.hpp
        SineSweepReader.hpp
        ProcessCalibrationData.hpp
        FineTuneModelGen.hpp
        LegacyMapLoader.hpp
        NPZMapLoader.hpp
    DESTINATION include/identification
)