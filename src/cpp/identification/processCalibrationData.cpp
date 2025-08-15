#include <pybind11/pybind11.h>

namespace py = pybind11;

void placeholder() {}

PYBIND11_MODULE(processCalibrationData, m) {
    m.doc() = "C++ bindings for processCalibrationData";
    m.def("placeholder", &placeholder);
}