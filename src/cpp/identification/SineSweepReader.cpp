#include <pybind11/pybind11.h>

namespace py = pybind11;

void placeholder() {}

PYBIND11_MODULE(SineSweepReader, m) {
    m.doc() = "C++ bindings for SineSweepReader";
    m.def("placeholder", &placeholder);
}