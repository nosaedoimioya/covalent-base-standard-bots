#include <pybind11/pybind11.h>

namespace py = pybind11;

void placeholder() {}

PYBIND11_MODULE(fineTuneModelGen, m) {
    m.doc() = "C++ bindings for fineTuneModelGen";
    m.def("placeholder", &placeholder);
}