#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../cpp/identification/SineSweepReader.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_sinesweepreader, m) {
    py::class_<SineSweepReader>(m, "SineSweepReader")
        .def(py::init<const std::string&, std::size_t, std::size_t,
                      const std::string&, const std::string&, std::size_t,
                      double, double, double, double, double, double, const std::string&,
                      double, double, int, std::size_t>())
        .def("get_calibration_maps", &SineSweepReader::get_calibration_maps)
        .def("reset_calibration_maps", &SineSweepReader::reset_calibration_maps)
        .def("load_csv", &SineSweepReader::load_csv)
        .def("load_npz", &SineSweepReader::load_npz)
        .def("load_npy", &SineSweepReader::load_npy);
}