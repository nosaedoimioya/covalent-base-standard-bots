#include "MapGeneration.hpp"
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>


PYBIND11_MODULE(MapGeneration, m) {
    py::class_<CalibrationMap>(m, "CalibrationMap")
        .def(py::init<int,int,int>())
        .def("num_positions", &CalibrationMap::num_positions)
        .def("axes_commanded", &CalibrationMap::axes_commanded)
        .def("num_joints", &CalibrationMap::num_joints)
        .def("generateCalibrationMap", &CalibrationMap::generateCalibrationMap,
            py::arg("robot_input"), py::arg("robot_output"),
            py::arg("input_Ts"), py::arg("output_Ts"),
            py::arg("max_freq_fit")=15.0,
            py::arg("gravity_comp")=false,
            py::arg("shift_store_position")=0)
        .def("generate_from_pose", &CalibrationMap::generate_from_pose,
            py::arg("q_cmd"),
            py::arg("tcp_acc"),
            py::arg("input_Ts"),
            py::arg("output_Ts"),
            py::arg("segments"),
            py::arg("freq_cmd"),
            py::arg("robot_model"),
            py::arg("V_angle")=0.0,
            py::arg("R_radius")=0.0,
            py::arg("axis_commanded")=1, // 1-based
            py::arg("q0")=std::vector<double>(),
            py::arg("max_freq_fit")=15.0,
            py::arg("gravity_comp")=false,
            py::arg("store_row")=0,
            py::arg("store_axis")=0) // 0-based
        // Expose matrices to Python
        .def_property_readonly("allWn",      [](const CalibrationMap& s){ return s.allWn(); })
        .def_property_readonly("allZeta",    [](const CalibrationMap& s){ return s.allZeta(); })
        .def_property_readonly("allWn2",     [](const CalibrationMap& s){ return s.allWn2(); })
        .def_property_readonly("allZeta2",   [](const CalibrationMap& s){ return s.allZeta2(); })
        .def_property_readonly("allInertia", [](const CalibrationMap& s){ return s.allInertia(); })
        .def_property_readonly("allV",       [](const CalibrationMap& s){ return s.allV(); })
        .def_property_readonly("allR",       [](const CalibrationMap& s){ return s.allR(); })
        .def("save_map", &CalibrationMap::save_map)
        .def_static("load_map", &CalibrationMap::load_map)
        .def("save_npz", &CalibrationMap::save_npz)
        .def_static("load_npz", &CalibrationMap::load_npz);
}