#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <Eigen/Dense>
#include "cnpy.h"

// SineSweepReader
// ---------------
// Lightweight C++ port of the Python SineSweepReader used for processing
// identification data.  The class focuses on file parsing and exposes
// utilities required by higher level routines.

class SineSweepReader {
public:
    SineSweepReader(const std::string& data_folder,
                    std::size_t num_poses,
                    std::size_t num_axes,
                    const std::string& robot_name,
                    const std::string& data_format,
                    std::size_t num_joints,
                    double min_freq,
                    double max_freq,
                    double freq_space,
                    double max_disp,
                    double dwell,
                    double Ts,
                    const std::string& ctrl_config,
                    double max_acc,
                    double max_vel,
                    int sine_cycles,
                    std::size_t max_map_size)
        : data_folder_(data_folder),
          num_poses_(num_poses),
          num_axes_(num_axes),
          robot_name_(robot_name),
          data_format_(data_format),
          num_joints_(num_joints),
          min_freq_(min_freq),
          max_freq_(max_freq),
          freq_space_(freq_space),
          max_disp_(max_disp),
          dwell_(dwell),
          Ts_(Ts),
          ctrl_config_(ctrl_config),
          max_acc_(max_acc),
          max_vel_(max_vel),
          sine_cycles_(sine_cycles),
          max_map_size_(max_map_size) {}

    // Returns the paths of generated calibration maps.  Real implementations
    // would perform the heavy lifting; for now this merely returns any
    // previously stored results.
    std::vector<std::string> get_calibration_maps() const { return calibration_maps_; }

    void reset_calibration_maps() { calibration_maps_.clear(); }

    // Helper routines for reading common data formats ----------------------
    Eigen::MatrixXd load_csv(const std::string& filename) const {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open CSV file: " + filename);
        }
        std::vector<double> values;
        std::size_t rows = 0;
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                values.push_back(std::stod(cell));
            }
            ++rows;
        }
        std::size_t cols = rows ? values.size() / rows : 0;
        Eigen::MatrixXd result(rows, cols);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                result(i, j) = values[i * cols + j];
            }
        }
        return result;
    }

    std::unordered_map<std::string, Eigen::MatrixXd>
    load_npz(const std::string& filename) const {
        cnpy::npz_t npz = cnpy::npz_load(filename);
        std::unordered_map<std::string, Eigen::MatrixXd> data;
        for (auto& kv : npz) {
            const std::string& name = kv.first;
            cnpy::NpyArray arr = kv.second;
            const double* raw = arr.data<double>();
            std::vector<std::size_t> shape(arr.shape.begin(), arr.shape.end());
            std::size_t rows = shape.empty() ? 0 : shape[0];
            std::size_t cols = shape.size() > 1 ? shape[1] : 1;
            Eigen::MatrixXd mat(rows, cols);
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < cols; ++j) {
                    mat(i, j) = raw[i * cols + j];
                }
            }
            data[name] = mat;
        }
        return data;
    }

    Eigen::MatrixXd load_npy(const std::string& filename) const {
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        const double* raw = arr.data<double>();
        std::vector<std::size_t> shape(arr.shape.begin(), arr.shape.end());
        std::size_t rows = shape.empty() ? 0 : shape[0];
        std::size_t cols = shape.size() > 1 ? shape[1] : 1;
        Eigen::MatrixXd mat(rows, cols);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                mat(i, j) = raw[i * cols + j];
            }
        }
        return mat;
    }

private:
    std::string data_folder_;
    std::size_t num_poses_;
    std::size_t num_axes_;
    std::string robot_name_;
    std::string data_format_;
    std::size_t num_joints_;
    double min_freq_;
    double max_freq_;
    double freq_space_;
    double max_disp_;
    double dwell_;
    double Ts_;
    std::string ctrl_config_;
    double max_acc_;
    double max_vel_;
    int sine_cycles_;
    std::size_t max_map_size_;

    std::vector<std::string> calibration_maps_;
};