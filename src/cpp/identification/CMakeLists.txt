set(IDENTIFICATION_DEPS)
if(Torch_FOUND)
    list(APPEND IDENTIFICATION_DEPS "${TORCH_LIBRARIES}")
endif()
if(Eigen3_FOUND)
    list(APPEND IDENTIFICATION_DEPS Eigen3::Eigen)
endif()
if(FFTW3_FOUND)
    list(APPEND IDENTIFICATION_DEPS FFTW3::fftw3)
endif()

if(pybind11_FOUND)
    pybind11_add_module(MapFitterDelete MODULE MapFitterDelete.cpp)
    target_link_libraries(MapFitterDelete PRIVATE ${IDENTIFICATION_DEPS})

    pybind11_add_module(MapGeneration MODULE MapGeneration.cpp)
    target_link_libraries(MapGeneration PRIVATE ${IDENTIFICATION_DEPS})

    pybind11_add_module(SineSweepReader MODULE SineSweepReader.cpp)
    target_link_libraries(SineSweepReader PRIVATE ${IDENTIFICATION_DEPS})

    pybind11_add_module(fineTuneModelGen MODULE fineTuneModelGen.cpp)
    target_link_libraries(fineTuneModelGen PRIVATE ${IDENTIFICATION_DEPS})

    pybind11_add_module(processCalibrationData MODULE processCalibrationData.cpp)
    target_link_libraries(processCalibrationData PRIVATE ${IDENTIFICATION_DEPS})

    pybind11_add_module(IDENTIFICATION_MODULE MODULE IDENTIFICATION_MODULE.cpp)
    target_link_libraries(IDENTIFICATION_MODULE PRIVATE ${IDENTIFICATION_DEPS})
else()
    message(STATUS "pybind11 not found; identification Python modules will not be built")
endif()

install(
    FILES
        MapFitter.hpp
        MapGeneration.hpp
        SineSweepReader.hpp
        processCalibrationData.hpp
        fineTuneModelGen.hpp
    DESTINATION include/identification
)