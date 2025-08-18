#include "FineTuneModelGen.hpp"
#include "MapGeneration.hpp"
#include "MapFitter.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tuple>
#include <vector>
#include <string>

namespace {

std::tuple<int, int, int> infer_dims(const std::vector<std::string> &map_files) {
    int total_poses = 0;
    int num_axes = 0;
    int num_joints = 0;
    for (const auto &file : map_files) {
        CalibrationMap m = CalibrationMap::load_map(file);
        total_poses += m.num_positions();
        if (num_axes == 0) {
            num_axes = m.axes_commanded();
            num_joints = m.num_joints();
        }
    }
     return {total_poses, num_axes, num_joints};
}

} // namespace

int runFineTuneModelGen(const std::string &model_file,
                        const std::vector<std::string> &maps,
                        int epochs,
                        double lr,
                        const std::string &save_file) {
    auto [num_poses, num_axes, num_joints] = infer_dims(maps);

    auto fitter = identification::ModelLoader::load(model_file, num_axes,
                                                   /*input_features*/3,
                                                   std::vector<int>{64, 64});

    // Placeholder: In a full implementation, data from the maps would be
    // converted into training tensors and passed to fitter->train(...). The
    // current version simply reloads and saves the models.
    fitter->save_models(save_file.empty() ? "fine_tuned_map" : save_file);

    return 0;
}

namespace py = pybind11;

PYBIND11_MODULE(FineTuneModelGen, m) {
    m.doc() = "C++ bindings for FineTuneModelGen";
    m.def("runFineTuneModelGen", &runFineTuneModelGen,
          py::arg("model_file"),
          py::arg("maps"),
          py::arg("epochs") = 50,
          py::arg("lr") = 1e-4,
          py::arg("save_file") = "");
}