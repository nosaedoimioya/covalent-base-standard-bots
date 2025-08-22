#include "MapFitter.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(MapFitter, m) {
    m.doc() = "Two-head torch model for calibration map fitting";

    py::class_<identification::MapFitter,
               std::shared_ptr<identification::MapFitter>>(m, "MapFitter")
        .def(py::init<int,int,std::vector<int>>(),
             py::arg("axes"), py::arg("input_features"), py::arg("hidden"))
        .def("train", &identification::MapFitter::train,
             py::arg("features"), py::arg("modes"), py::arg("orders"), py::arg("masks"),
             py::arg("epochs") = 200, py::arg("lr") = 1e-3)
        .def("infer", &identification::MapFitter::infer,
             py::arg("axis"), py::arg("feature"))
        .def("save_models", &identification::MapFitter::save_models,
             py::arg("directory"))
        .def("load_models", &identification::MapFitter::load_models,
             py::arg("directory"));

    py::class_<identification::ModelLoader>(m, "ModelLoader")
        .def_static("load", &identification::ModelLoader::load,
                    py::arg("directory"), py::arg("axes"),
                    py::arg("input_features"), py::arg("hidden"));
}
