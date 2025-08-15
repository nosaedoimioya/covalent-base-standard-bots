#include <pybind11/pybind11.h>

namespace py = pybind11;

void placeholder() {}

PYBIND11_MODULE(MapFitter, m) {
    m.doc() = "C++ bindings for MapFitter";
    m.def("placeholder", &placeholder);
}