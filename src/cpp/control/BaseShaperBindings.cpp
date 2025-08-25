#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "BaseShaper.hpp"

namespace py = pybind11;

PYBIND11_MODULE(base_shaper, m) {
    m.doc() = "Python bindings for BaseShaper C++ class"; // Optional module docstring

    py::class_<control::BaseShaper>(m, "BaseShaper")
        .def(py::init<double>(), py::arg("Ts"), 
             "Initialize BaseShaper with sampling time Ts")
        .def("shape_sample", &control::BaseShaper::shape_sample, 
             py::arg("x_i"), py::arg("frf_params"),
             "Shape a single input sample using current dynamics parameters")
        .def("shape_trajectory", &control::BaseShaper::shape_trajectory,
             py::arg("x"), py::arg("varying_params"),
             "Shape a complete trajectory using varying dynamics parameters")
        .def("compute_zvd_shaper", &control::BaseShaper::compute_zvd_shaper,
             py::arg("params_array"),
             "Compute ZVD shaper impulse response for given dynamics parameters");
}
