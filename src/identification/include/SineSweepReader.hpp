#pragma once


#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#include <Eigen/Dense>
#include "cnpy.h"

/**
 * @file SineSweepReader.hpp
 * @brief Lightweight C++ port of the Python SineSweepReader used for
 *        processing identification data.
 * 
 * This module provides comprehensive utilities for reading and processing sine sweep
 * identification data from various file formats (CSV, NPY, NPZ). It handles the
 * complete pipeline from raw data loading to calibration map generation, supporting
 * multiple robot configurations and data formats commonly used in robot dynamics
 * identification experiments.
 */

/**
 * @brief Parses sine sweep identification data and exposes helper utilities.
 * 
 * SineSweepReader is the core class for processing robot identification experiments
 * that use sine sweep excitation signals. It manages the complete data processing
 * pipeline including file I/O, data format conversion, and calibration map generation.
 * The class supports multiple input formats and provides utilities for handling
 * different robot configurations and experimental setups.
 */
class SineSweepReader {
public:
    /**
     * @brief Construct a reader for sine sweep datasets.
     * 
     * Initializes a SineSweepReader with all necessary parameters for processing
     * sine sweep identification data. The constructor validates parameters and
     * prepares the reader for data processing operations.
     * 
     * @param data_folder   Root directory containing the identification data.
     * @param num_poses     Number of pose samples in the dataset.
     * @param num_axes      Number of commanded axes during identification.
     * @param robot_name    Name/identifier of the robot being identified.
     * @param data_format   Input data format ("csv", "npy", "npz").
     * @param num_joints    Number of joints in the robot.
     * @param min_freq      Minimum excitation frequency in Hz.
     * @param max_freq      Maximum excitation frequency in Hz.
     * @param freq_space    Frequency spacing between excitation points.
     * @param max_disp      Maximum displacement amplitude in radians.
     * @param dwell         Dwell time between frequency sweeps in seconds.
     * @param Ts            Sample period in seconds.
     * @param ctrl_config   Control configuration identifier.
     * @param max_acc       Maximum acceleration limit in rad/s².
     * @param max_vel       Maximum velocity limit in rad/s.
     * @param sine_cycles   Number of sine cycles per frequency point.
     * @param max_map_size  Maximum number of poses per calibration map.
     */
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

    /**
     * @brief Retrieve paths of generated calibration maps.
     * 
     * Returns the file paths of all calibration maps that have been generated
     * or discovered by this reader. These maps contain the processed dynamics
     * parameters extracted from the sine sweep data.
     * 
     * @return Vector of map file paths.
     */
    std::vector<std::string> get_calibration_maps();
    
    /**
     * @brief Clear any stored calibration map paths.
     * 
     * Resets the internal cache of calibration map paths. This is useful when
     * reprocessing data or when the map generation process needs to be restarted.
     */
    void reset_calibration_maps();

    // Helper routines for reading common data formats ----------------------

    /**
     * @brief Compute how many calibration maps are required.
     *
     * Calculates the number of calibration maps needed based on the total number
     * of poses and the maximum allowed map size. This mirrors the logic of the
     * legacy Python implementation which divides the total poses by the maximum
     * map size and rounds up with tolerance for floating point error.
     * 
     * @return Number of calibration maps required to store all pose data.
     */
    std::size_t compute_num_maps() const;

    /**
     * @brief Parse a simple delimited text file into numeric rows.
     *
     * Reads a text file and converts each non-empty line into a vector of doubles
     * by splitting on commas. This routine is intentionally lightweight and is used
     * by tests to validate data handling parity with the historic Python implementation.
     * The function handles basic CSV parsing with minimal overhead.
     *
     * @param filename Path to the data file to parse.
     * @return Parsed matrix stored as a vector of rows, where each row is a vector of doubles.
     */
    std::vector<std::vector<double>> parse_data(const std::string& filename) const;

    /**
     * @brief Load a CSV file into an Eigen matrix.
     * 
     * Parses a comma-separated values file and converts it into an Eigen MatrixXd
     * for efficient numerical operations. The function handles basic CSV format
     * with automatic dimension detection based on the data.
     * 
     * @param filename Path to the CSV file to load.
     * @return Parsed matrix as Eigen::MatrixXd.
     * @throws std::runtime_error if the file cannot be opened or parsed.
     */
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

    /**
     * @brief Load an NPZ archive into a map of Eigen matrices.
     * 
     * Reads a NumPy compressed archive (.npz file) and extracts all arrays
     * into a map where keys are array names and values are Eigen matrices.
     * This is useful for loading complex datasets with multiple named arrays.
     * 
     * @param filename Path to the NPZ file to load.
     * @return Mapping from array names to corresponding Eigen matrices.
     */
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

    /**
     * @brief Load a NumPy NPY array file.
     * 
     * Reads a single NumPy array file (.npy) and converts it to an Eigen matrix.
     * This function handles the binary NumPy format and automatically detects
     * the array dimensions from the file header.
     * 
     * @param filename Path to the NPY file to load.
     * @return Parsed matrix as Eigen::MatrixXd.
     */
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
    std::string data_folder_;      ///< Root directory containing identification data
    std::size_t num_poses_;        ///< Number of pose samples in the dataset
    std::size_t num_axes_;         ///< Number of commanded axes during identification
    std::string robot_name_;       ///< Name/identifier of the robot
    std::string data_format_;      ///< Input data format (csv, npy, npz)
    std::size_t num_joints_;       ///< Number of joints in the robot
    double min_freq_;              ///< Minimum excitation frequency in Hz
    double max_freq_;              ///< Maximum excitation frequency in Hz
    double freq_space_;            ///< Frequency spacing between excitation points
    double max_disp_;              ///< Maximum displacement amplitude in radians
    double dwell_;                 ///< Dwell time between frequency sweeps in seconds
    double Ts_;                    ///< Sample period in seconds
    std::string ctrl_config_;      ///< Control configuration identifier
    double max_acc_;               ///< Maximum acceleration limit in rad/s²
    double max_vel_;               ///< Maximum velocity limit in rad/s
    int sine_cycles_;              ///< Number of sine cycles per frequency point
    std::size_t max_map_size_;     ///< Maximum number of poses per calibration map

    std::vector<std::string> calibration_maps_;  ///< Cache of generated calibration map paths

    /**
     * @brief Read file contents into a vector of strings.
     * 
     * Helper function to read a text file line by line into a vector of strings.
     * Used internally for parsing configuration files and other text-based data.
     * 
     * @param filename Path to the file to read.
     * @return Vector of strings, one per line in the file.
     */
    std::vector<std::string> read_file(const std::string& filename) const;
};