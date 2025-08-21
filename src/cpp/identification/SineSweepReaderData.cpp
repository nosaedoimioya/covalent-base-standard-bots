#include "SineSweepReader.hpp"
#include "MapGeneration.hpp"
#include <sstream>
#include <filesystem>
#include <cmath>
#include <algorithm>

#include <cnpy.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
namespace fs = std::filesystem;

// Helper: robustly read the first row of a CSV and return as std::vector<double>
static std::vector<double> read_first_row_csv(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Unable to open static CSV: " + filename);
    }
    std::string line;
    while (std::getline(f, line)) {
        // skip empty/whitespace lines
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty())
                row.push_back(std::stod(cell));
        }
        return row; // first non-empty line only
    }
    throw std::runtime_error("Static CSV has no data rows: " + filename);
}
// ---------- Fallback C++ loaders (used only if Python util import fails) -----

static Eigen::MatrixXd load_csv_numeric_body(const std::string& fn) {
    std::ifstream file(fn);
    if (!file.is_open()) throw std::runtime_error("Unable to open CSV: " + fn);
    std::string line;
    std::vector<std::vector<double>> rows;
    bool first = true;
    while (std::getline(file, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        bool non_numeric = false;
        while (std::getline(ss, cell, ',')) {
            try {
                if (!cell.empty()) row.push_back(std::stod(cell));
            } catch (...) {
                non_numeric = true;
                break;
            }
        }
        // skip header row
        if (first && non_numeric) { first = false; continue; }
        first = false;
        if (!non_numeric && !row.empty()) rows.push_back(std::move(row));
    }
    if (rows.empty()) return Eigen::MatrixXd();
    const Eigen::Index R = static_cast<Eigen::Index>(rows.size());
    const Eigen::Index C = static_cast<Eigen::Index>(rows.front().size());
    Eigen::MatrixXd M(R, C);
    for (Eigen::Index i=0;i<R;++i){
        if (static_cast<Eigen::Index>(rows[i].size()) != C)
            throw std::runtime_error("CSV column mismatch (fallback loader): " + fn);
        for (Eigen::Index j=0;j<C;++j) M(i,j) = rows[i][j];
    }
    return M;
}

static Eigen::MatrixXd load_npy_2d(const std::string& fn) {
    cnpy::NpyArray arr = cnpy::npy_load(fn);
    if (arr.shape.size() != 2) throw std::runtime_error("NPY must be 2D: " + fn);
    const size_t R = arr.shape[0], C = arr.shape[1];
    const double* p = arr.data<double>();
    Eigen::MatrixXd M(R, C);
    for (size_t i=0;i<R;++i)
        for (size_t j=0;j<C;++j)
            M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = p[i*C + j];
    return M;
}

static Eigen::MatrixXd load_npz_first_2d(const std::string& fn) {
    cnpy::npz_t z = cnpy::npz_load(fn);
    if (z.empty()) throw std::runtime_error("NPZ has no arrays: " + fn);
    auto it = z.begin();
    cnpy::NpyArray arr = it->second;
    if (arr.shape.size() != 2) throw std::runtime_error("NPZ array not 2D: " + fn);
    const size_t R = arr.shape[0], C = arr.shape[1];
    const double* p = arr.data<double>();
    Eigen::MatrixXd M(R, C);
    for (size_t i=0;i<R;++i)
        for (size_t j=0;j<C;++j)
            M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = p[i*C + j];
    return M;
}

static Eigen::MatrixXd load_matrix_fallback(const std::string& filename) {
    std::string ext;
    if (auto pos = filename.find_last_of('.'); pos != std::string::npos) ext = filename.substr(pos+1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "csv") return load_csv_numeric_body(filename);
    if (ext == "npy") return load_npy_2d(filename);
    if (ext == "npz") return load_npz_first_2d(filename);
    throw std::runtime_error("Unknown extension for: " + filename);
}

// ----------------- Segment detection (like Python correlate_sine_wave) ------

static std::pair<int,int> correlate_sine(const Eigen::VectorXd& input_signal,
                                         double freq,
                                         double Ts,
                                         int sine_cycles,
                                         double max_disp,
                                         double max_acc) {
    const int Ns = static_cast<int>(std::ceil(((1.0/freq)*sine_cycles)/Ts));
    Eigen::VectorXd t(Ns);
    for (int j=0;j<Ns;++j) t(j) = j * Ts;

    const double sin_amplitude = std::min(2.0*M_PI*freq*std::sqrt(max_disp), max_acc);
    Eigen::VectorXd q = -(sin_amplitude / std::pow(2.0*M_PI*freq,2)) * t.array().sin().matrix();

    const int Nx = static_cast<int>(input_signal.size());
    const int Ny = static_cast<int>(q.size());
    const int Ncorr = Nx + Ny - 1;
    Eigen::VectorXd corr = Eigen::VectorXd::Zero(Ncorr);

    for (int n=0;n<Ncorr;++n){
        double acc = 0.0;
        for (int k=0;k<Nx;++k){
            int yidx = n - k;
            if (yidx >= 0 && yidx < Ny) acc += input_signal(k) * q(yidx);
        }
        corr(n) = acc;
    }

    Eigen::Index maxIdx;
    corr.maxCoeff(&maxIdx);

    const int lag = static_cast<int>(maxIdx) - (Ny - 1);
    const int f_start = std::max(0, lag+1);
    const int f_end   = std::min(Nx-1, f_start + Ny - 1);

    return {f_start, f_end};
}

std::size_t SineSweepReader::compute_num_maps() const {
    if (max_map_size_ == 0) {
        return 1;  // avoid division by zero; treat as single map
    }
    double runs = static_cast<double>(num_poses_) /
                  static_cast<double>(max_map_size_);
    std::size_t whole = static_cast<std::size_t>(runs);
    const double tol = 1e-6;
    if (runs - static_cast<double>(whole) > tol) {
        return whole + 1;
    }
    return whole;
}

static py::object make_robot_model_from_name(const std::string& robot_name) {
    // 1) Make sure Python can import your modules: add "<repo>/src" to sys.path if needed.
    py::module sys = py::module::import("sys");
    py::list sys_path = sys.attr("path");

    // Try current working dir upward to locate "src/util/RobotInterface.py"
    fs::path here = fs::current_path();
    bool inserted = false;
    for (int up = 0; up <= 5 && !inserted; ++up) {
        fs::path base = here;
        for (int i = 0; i < up; ++i) base = base.parent_path();
        fs::path candidate = base / "src" / "util" / "RobotInterface.py";
        if (fs::exists(candidate)) {
            sys_path.attr("insert")(0, (base / "src").string());
            inserted = true;
        }
    }

    // 2) Import RobotInterface and build the robot by name (always sim for map gen)
    py::object RobotInterface = py::module::import("util.RobotInterface").attr("RobotInterface");
    py::object iface = RobotInterface(robot_name, "sim");  // kwargs: (robot_name, robot_ip="sim")
    // 3) Get the robot model (util.RobotDynamics.Dynamics)
    py::object model = iface.attr("robot").attr("model");
    return model;
}

std::vector<std::string> SineSweepReader::get_calibration_maps() {
    reset_calibration_maps();
    std::size_t runs = compute_num_maps();

    // Build frequency list once
    std::vector<double> freq_range;
    for (double f = min_freq_; f <= max_freq_ + 1e-9; f += freq_space_) freq_range.push_back(f);
    if (freq_range.empty())
        throw std::runtime_error("Frequency range is empty; check min/max/spacing.");

    // Create Python Dynamics model once (you can also swap to RobotInterface if you prefer)
    py::object robot_model = make_robot_model_from_name(robot_name_);

    for (std::size_t pass = 0; pass < runs; ++pass) {
        const std::size_t start_pose = pass * max_map_size_;
        const std::size_t last_pose  = (max_map_size_ == 0) ? num_poses_ : std::min(num_poses_, (pass + 1) * max_map_size_);
        const std::size_t batch_size = last_pose - start_pose;

        if (batch_size == 0) continue;

        // Create the map container for this batch
        CalibrationMap cmap(static_cast<int>(batch_size),
                            static_cast<int>(num_axes_),
                            static_cast<int>(num_joints_));


        for (std::size_t p = 0; p < batch_size; ++p) {
            const std::size_t pose = start_pose + p;

            // Load per-axis motion; build segments via correlation; call generatePoint
            for (std::size_t a = 0; a < num_axes_; ++a) {
                const std::string dyn_file =
                    data_folder_ + "/robotData_motion_pose" + std::to_string(pose) +
                    "_axis" + std::to_string(a) + "." + data_format_;

                // Use Python Utility loader if available; fallback to C++
                Eigen::MatrixXd D;
                try {
                    py::object util = py::module::import("util.Utility");
                    py::tuple res   = util.attr("load_data_file")(data_format_, dyn_file).cast<py::tuple>();
                    D = res[0].cast<Eigen::MatrixXd>();
                } catch (...) {
                    D = load_matrix_fallback(dyn_file);
                }

                // column layout same as your Python reader
                const int TIME_IDX = 2;
                const int cmd_start = TIME_IDX + 1;
                const int cmd_end   = cmd_start + static_cast<int>(num_joints_);
                const int cur_end   = cmd_end + static_cast<int>(num_joints_) * 2; // enc+currents
                const int acc_t_idx = cur_end;
                const int acc_start = acc_t_idx + 1;  // x,y,z
                const int acc_end   = acc_start + 3;

                if (D.cols() < acc_end) continue;

                // time zeroed
                Eigen::VectorXd time = D.col(TIME_IDX).array();
                time.array() -= D(0, TIME_IDX);
                // commanded joints [N x J]
                Eigen::MatrixXd qcmd = D.block(0, cmd_start, D.rows(), num_joints_);
                // tcp accel [N x 3]
                Eigen::MatrixXd acc  = D.block(0, acc_start, D.rows(), 3);

                // initial joint angles
                std::vector<double> init_q(num_joints_);
                for (int j=0;j<num_joints_;++j) init_q[j] = qcmd(0,j);

                // ---- segment detection (same as Python correlate_sine_wave)
                std::vector<std::pair<int,int>> begin_end;
                std::vector<double>              freqs;
                // temp signal: delta joint position (this axis)
                Eigen::VectorXd INPUT_temp = qcmd.col(static_cast<int>(a));
                INPUT_temp.array() -= qcmd(0, static_cast<int>(a));
                int f_start_init = 0;

                for (double f : freq_range) {
                    auto seg = correlate_sine(INPUT_temp, f, Ts_, sine_cycles_, max_disp_, max_acc_);
                    int fs = std::max(seg.first,  f_start_init);
                    int fe = std::min(seg.second, static_cast<int>(INPUT_temp.size())-1);
                    if (fe <= fs+2) continue;

                    // zero-out past used samples to avoid re-hits (Python behavior)
                    for (int k=1;k<fs && k<INPUT_temp.size();++k) INPUT_temp(k) = 0.0;
                    f_start_init = fs;

                    begin_end.emplace_back(fs, fe);
                    freqs.push_back(f);
                }
                if (begin_end.empty()) continue;

                // ---- V (angle) and R (radius) from static CSV (as in Python)
                double V_angle = 0.0, R_radius = 0.0;
                try {
                    const std::string static_file =
                        data_folder_ + "/robotData_static_pose" + std::to_string(pose) +
                        "_axis" + std::to_string(a) + ".csv";
                    py::object util = py::module::import("util.Utility");
                    py::tuple res   = util.attr("load_data_file")("csv", static_file).cast<py::tuple>();
                    Eigen::MatrixXd S = res[0].cast<Eigen::MatrixXd>();
                    if (S.rows() > 0 && S.cols() >= 2) {
                        V_angle = S(0,0);
                        R_radius = S(0,1);
                    }
                } catch (...) {
                    // leave zeros if missing
                }

                // ---- Call into MapGeneration heavy routine for this (pose, axis)
                cmap.generate_from_pose(qcmd,              // [N x J]
                                        acc,               // [N x 3]
                                        Ts_, Ts_,
                                        begin_end,
                                        freqs,
                                        robot_model,                 // Dynamics object
                                        V_angle, R_radius,
                                        static_cast<int>(a) + 1,     // 1-based
                                        init_q,
                                        max_freq_,
                                        /*gravity_comp=*/true,
                                        static_cast<int>(p),         // row in map
                                        static_cast<int>(a)         // col in map
                                        );
            } // axis loop
        } // pose loop
        
        std::ostringstream fn;
        fn << data_folder_ << "/" << robot_name_
           << "_robot_calibration_map_lastPose" << last_pose
           << "_numAxes" << num_axes_
           << "_startPose" << start_pose << ".npz";
        
        // ensure directory exists
        fs::create_directories(fs::path(fn.str()).parent_path());
        cmap.save_npz(fn.str());
        calibration_maps_.push_back(fn.str());
    }

    return calibration_maps_;
}

std::vector<std::string> SineSweepReader::read_file(
    const std::string& filename) const {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() &&
            line.find_first_not_of(" \t\r\n") != std::string::npos) {
            lines.push_back(line);
        }
    }
    return lines;
}

std::vector<std::vector<double>> SineSweepReader::parse_data(
    const std::string& filename) const {
    std::vector<std::string> lines = read_file(filename);
    std::vector<std::vector<double>> parsed;
    for (const std::string& line : lines) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                row.push_back(std::stod(cell));
            }
        }
        if (!row.empty()) {
            parsed.push_back(std::move(row));
        }
    }
    return parsed;
}