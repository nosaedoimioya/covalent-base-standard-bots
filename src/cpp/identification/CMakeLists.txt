set(IDENTIFICATION_DEPS)
if(Torch_FOUND)
    list(APPEND IDENTIFICATION_DEPS "${TORCH_LIBRARIES}")
endif()
if(Eigen3_FOUND)
    list(APPEND IDENTIFICATION_DEPS Eigen3::Eigen)
endif()
if(Iir_FOUND)
    list(APPEND IDENTIFICATION_DEPS Iir::iir)
endif()
if(cnpy_FOUND)
    list(APPEND IDENTIFICATION_DEPS cnpy::cnpy)
endif()

if(pybind11_FOUND)
    pybind11_add_module(MapFitter MODULE MapFitter.cpp)
    target_link_libraries(MapFitter PRIVATE ${IDENTIFICATION_DEPS})

    pybind11_add_module(MapGeneration MODULE MapGeneration.cpp)
    target_link_libraries(MapGeneration PRIVATE ${IDENTIFICATION_DEPS} FFTW3::fftw3)

    pybind11_add_module(SineSweepReader MODULE SineSweepReader.cpp)
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