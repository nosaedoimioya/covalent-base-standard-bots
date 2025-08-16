#include "SineSweepReader.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Bindings for the lightweight ``SineSweepReader`` class.  The module name
// matches the original Python extension for compatibility with the existing
// stub in ``identification/SineSweepReader.py``.
PYBIND11_MODULE(SineSweepReader, m) {
    py::class_<SineSweepReader>(m, "SineSweepReader")
        .def(py::init<const std::string&, std::size_t, std::size_t,
                      const std::string&, const std::string&, std::size_t,
                      double, double, double, double, double, double,
                      const std::string&, double, double, int, std::size_t>())
        .def("get_calibration_maps", &SineSweepReader::get_calibration_maps)
        .def("reset_calibration_maps", &SineSweepReader::reset_calibration_maps)
        .def("compute_num_maps", &SineSweepReader::compute_num_maps)
        .def("parse_data", &SineSweepReader::parse_data)
        .def("load_csv", &SineSweepReader::load_csv)
        .def("load_npz", &SineSweepReader::load_npz)
        .def("load_npy", &SineSweepReader::load_npy);
}