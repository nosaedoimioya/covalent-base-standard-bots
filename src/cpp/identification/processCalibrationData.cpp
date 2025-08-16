#include "ProcessCalibrationData.hpp"
#include "MapFitter.hpp"
#include "SineSweepReader.hpp"
#include "CLI11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

int processCalibrationData(const ProcessCalibrationOptions &in_opts) {
    ProcessCalibrationOptions opts = in_opts;
    if (opts.poses <= 0) {
        throw std::runtime_error("'poses' must be > 0");
    }
    if (opts.axes <= 0) {
        throw std::runtime_error("'axes' must be > 0");
    }
    if (opts.min_freq < 0 || opts.max_freq < opts.min_freq) {
        throw std::runtime_error("frequency range invalid (0 ≤ minfreq ≤ maxfreq)");
    }
    if (opts.max_map_size <= 0) {
        throw std::runtime_error("'max-map-size' must be > 0");
    }
    if (opts.max_map_size > opts.poses) {
        opts.max_map_size = opts.poses;
    }

    std::vector<std::string> stored_maps;
    if (opts.saved_maps) {
        int start = opts.start_pose;
        while (start < opts.poses) {
            int last = std::min(start + opts.max_map_size, opts.poses);
            std::ostringstream fn;
            fn << opts.data_path << "/" << opts.robot_name
               << "_robot_calibration_map_lastPose" << last
               << "_numAxes" << opts.axes
               << "_startPose" << start << ".pkl";
            if (!fs::exists(fn.str())) {
                throw std::runtime_error("Calibration map file " + fn.str() + " doesn't exist.");
            }
            stored_maps.push_back(fn.str());
            start = last;
        }
    } else {
        SineSweepReader reader(opts.data_path, opts.poses, opts.axes,
                               opts.robot_name, opts.file_format,
                               opts.num_joints, opts.min_freq, opts.max_freq,
                               opts.freq_space, opts.max_disp, opts.dwell,
                               opts.Ts, opts.ctrl_config, opts.max_acc,
                               opts.max_vel, opts.sine_cycles, opts.max_map_size);
        stored_maps = reader.get_calibration_maps();
    }

    // Integrate MapFitter: in this minimal pipeline we simply create the
    // object and save placeholder models.
    identification::MapFitter fitter(opts.axes, 1, {8});
    std::string model_dir = opts.data_path + "/models";
    fitter.save_models(model_dir);

    return static_cast<int>(stored_maps.size());
}

int processCalibrationDataCLI(int argc, const char **argv) {
    ProcessCalibrationOptions opts;
    CLI::App app{"Load sysID sine-sweep data and produce calibration maps"};
    app.add_option("data_path", opts.data_path,
                   "Path to data folder with sysID files")->required();
    app.add_option("poses", opts.poses,
                   "Number of poses robot visits")->required();
    app.add_option("axes", opts.axes,
                   "Number of commanded axes per pose")->required();
    app.add_option("--name", opts.robot_name, "Robot name")
        ->default_val(opts.robot_name);
    app.add_option("--format", opts.file_format, "Data file format")
        ->default_val(opts.file_format)
        ->check(CLI::IsMember({"csv", "npz", "npy"}));
    app.add_option("--numjoints", opts.num_joints,
                   "Number of robot joints")
        ->default_val(opts.num_joints);
    app.add_option("--minfreq", opts.min_freq, "Min frequency [Hz]")
        ->default_val(opts.min_freq);
    app.add_option("--maxfreq", opts.max_freq, "Max frequency [Hz]")
        ->default_val(opts.max_freq);
    app.add_option("--freqspace", opts.freq_space,
                   "Frequency spacing [Hz]")
        ->default_val(opts.freq_space);
    app.add_option("--mdisp", opts.max_disp, "Max sweep stroke [rad]")
        ->default_val(opts.max_disp);
    app.add_option("--dwell", opts.dwell, "Post-sweep dwell [s]")
        ->default_val(opts.dwell);
    app.add_option("--timestep", opts.Ts, "Sampling time [s]")
        ->default_val(opts.Ts);
    app.add_option("--type", opts.sysid_type, "SysID type")
        ->default_val(opts.sysid_type)
        ->check(CLI::IsMember({"bcb", "sine"}));
    app.add_option("--ctrl", opts.ctrl_config, "Control mode")
        ->default_val(opts.ctrl_config)
        ->check(CLI::IsMember({"joint", "task"}));
    app.add_option("--macc", opts.max_acc,
                   "Max acceleration [rad/s^2]")
        ->default_val(opts.max_acc);
    app.add_option("--mvel", opts.max_vel, "Max velocity [rad/s]")
        ->default_val(opts.max_vel);
    app.add_option("--sine-cycles", opts.sine_cycles,
                   "Number of sine cycles per pose")
        ->default_val(opts.sine_cycles);
    app.add_option("--sensor", opts.sensor, "Sensor type")
        ->default_val(opts.sensor)
        ->check(CLI::IsMember({"ToolAcc", "JointPos"}));
    app.add_option("--first-pose", opts.start_pose,
                   "Starting pose index")
        ->default_val(opts.start_pose);
    app.add_option("--max-map-size", opts.max_map_size,
                   "Max poses per map")
        ->default_val(opts.max_map_size);
    app.add_flag("--saved-maps", opts.saved_maps,
                 "Load existing calibration maps instead of generating new ones");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    return processCalibrationData(opts);
}

namespace py = pybind11;

PYBIND11_MODULE(ProcessCalibrationData, m) {
    m.doc() = "C++ bindings for ProcessCalibrationData";
    m.def(
        "processCalibrationData",
        [](const std::string &data_path, int poses, int axes,
           const std::string &robot_name, const std::string &file_format,
           int num_joints, double min_freq, double max_freq,
           double freq_space, double max_disp, double dwell, double Ts,
           const std::string &sysid_type, const std::string &ctrl_config,
           double max_acc, double max_vel, int sine_cycles,
           const std::string &sensor, int start_pose, int max_map_size,
           bool saved_maps) {
            ProcessCalibrationOptions opts;
            opts.data_path = data_path;
            opts.poses = poses;
            opts.axes = axes;
            opts.robot_name = robot_name;
            opts.file_format = file_format;
            opts.num_joints = num_joints;
            opts.min_freq = min_freq;
            opts.max_freq = max_freq;
            opts.freq_space = freq_space;
            opts.max_disp = max_disp;
            opts.dwell = dwell;
            opts.Ts = Ts;
            opts.sysid_type = sysid_type;
            opts.ctrl_config = ctrl_config;
            opts.max_acc = max_acc;
            opts.max_vel = max_vel;
            opts.sine_cycles = sine_cycles;
            opts.sensor = sensor;
            opts.start_pose = start_pose;
            opts.max_map_size = max_map_size;
            opts.saved_maps = saved_maps;
            return processCalibrationData(opts);
        },
        py::arg("data_path"), py::arg("poses"), py::arg("axes"),
        py::arg("robot_name") = ProcessCalibrationOptions().robot_name,
        py::arg("file_format") = ProcessCalibrationOptions().file_format,
        py::arg("num_joints") = ProcessCalibrationOptions().num_joints,
        py::arg("min_freq") = ProcessCalibrationOptions().min_freq,
        py::arg("max_freq") = ProcessCalibrationOptions().max_freq,
        py::arg("freq_space") = ProcessCalibrationOptions().freq_space,
        py::arg("max_disp") = ProcessCalibrationOptions().max_disp,
        py::arg("dwell") = ProcessCalibrationOptions().dwell,
        py::arg("Ts") = ProcessCalibrationOptions().Ts,
        py::arg("sysid_type") = ProcessCalibrationOptions().sysid_type,
        py::arg("ctrl_config") = ProcessCalibrationOptions().ctrl_config,
        py::arg("max_acc") = ProcessCalibrationOptions().max_acc,
        py::arg("max_vel") = ProcessCalibrationOptions().max_vel,
        py::arg("sine_cycles") = ProcessCalibrationOptions().sine_cycles,
        py::arg("sensor") = ProcessCalibrationOptions().sensor,
        py::arg("start_pose") = ProcessCalibrationOptions().start_pose,
        py::arg("max_map_size") = ProcessCalibrationOptions().max_map_size,
        py::arg("saved_maps") = ProcessCalibrationOptions().saved_maps);
    m.def("processCalibrationDataCLI", &processCalibrationDataCLI,
          py::arg("argv"));
}