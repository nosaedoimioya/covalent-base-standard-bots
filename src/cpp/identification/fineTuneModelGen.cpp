// FineTuneModelGen.cpp

#include "FineTuneModelGen.hpp"

#include "MapGeneration.hpp"      // CalibrationMap::load_map / load_npz
#include "MapFitter.hpp"          // identification::{MapFitter, ModelLoader}
#include "LegacyMapLoader.hpp"    // LoadLegacyTensorsFromPickle(...)
#include "NPZMapLoader.hpp"       // LoadNPZTensorsFromNPZ(...)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

// ---- helpers --------------------------------------------------------------

namespace {

// Determine extension of the first map (we require all maps to match).
inline std::string first_ext(const std::vector<std::string> &files) {
    if (files.empty()) throw std::runtime_error("No map files provided.");
    std::string ext = fs::path(files.front()).extension().string();
    for (const auto &f : files) {
        if (fs::path(f).extension().string() != ext) {
            throw std::runtime_error("All map files must share the same extension.");
        }
    }
    return ext;
}

// Infer (total poses, axes, joints) from the given maps.
// Supports both legacy .pkl and .npz containers.
std::tuple<int,int,int> infer_dims(const std::vector<std::string> &map_files) {
    if (map_files.empty()) return {0,0,0};

    const std::string ext = first_ext(map_files);
    int total_poses = 0;
    int num_axes = 0;
    int num_joints = 0;

    for (const auto &file : map_files) {
        CalibrationMap m = (ext == ".npz")
            ? CalibrationMap::load_npz(file)
            : CalibrationMap::load_map(file);

        total_poses += m.num_positions();
        if (num_axes == 0) {
            num_axes   = m.axes_commanded();
            num_joints = m.num_joints();
        }
    }
    return {total_poses, num_axes, num_joints};
}

} // namespace

// ---- main API -------------------------------------------------------------

int runFineTuneModelGen(const std::string &model_file,
                        const std::vector<std::string> &maps,
                        int epochs,
                        double lr,
                        const std::string &save_file) {
    if (maps.empty()) {
        throw std::runtime_error("runFineTuneModelGen: 'maps' list is empty.");
    }
    if (!fs::exists(model_file)) {
        throw std::runtime_error("runFineTuneModelGen: model directory '" + model_file + "' not found.");
    }

    // Infer dimensions from the maps (matches Python behavior).
    auto [num_poses, num_axes, num_joints] = infer_dims(maps);
    if (num_axes <= 0) {
        throw std::runtime_error("Failed to infer axes/joints from provided maps.");
    }

    // Load existing models (two-head net per axis).
    auto fitter = identification::ModelLoader::load(
        model_file,
        /*axes=*/num_axes,
        /*input_features=*/3,      // [V_deg, R_mm, inertia]
        /*hidden=*/{64, 64}
    );

    // Convert maps to training tensors and fine-tune.
    const std::string ext = first_ext(maps);
    if (ext == ".pkl") {
        LegacyTensors T = LoadLegacyTensorsFromPickle(maps, num_axes);
        fitter->train(T.features, T.modes, T.orders, T.masks, epochs, lr);
    } else if (ext == ".npz") {
        auto T = LoadNPZTensorsFromNPZ(maps, num_axes);
        fitter->train(T.features, T.modes, T.orders, T.masks, epochs, lr);
    } else {
        throw std::runtime_error("Unsupported map extension '" + ext + "'. Expected .pkl or .npz.");
    }

    // Save updated models (do not overwrite unless caller points to same dir).
    std::string outdir = save_file.empty() ? std::string("fine_tuned_map") : save_file;
    fs::create_directories(outdir);
    fitter->save_models(outdir);

    (void)num_poses; (void)num_joints; // currently not used beyond sanity

    return 0;
}

// ---- pybind ---------------------------------------------------------------

namespace py = pybind11;

PYBIND11_MODULE(FineTuneModelGen, m) {
    m.doc() = "C++ bindings for FineTuneModelGen (fine-tune saved shaper NN models)";
    m.def("runFineTuneModelGen", &runFineTuneModelGen,
          py::arg("model_file"),
          py::arg("maps"),
          py::arg("epochs") = 50,
          py::arg("lr") = 1e-4,
          py::arg("save_file") = "");
}
