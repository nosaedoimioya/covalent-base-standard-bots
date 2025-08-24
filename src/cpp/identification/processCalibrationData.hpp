#pragma once

#include <cmath>
#include <string>

/**
 * @file ProcessCalibrationData.hpp
 * @brief Interfaces for converting raw identification data into calibration maps.
 * 
 * This module provides the main entry point for processing robot identification
 * experiments. It includes command-line interfaces and configuration structures
 * for converting raw sine sweep data into calibrated dynamics models. The module
 * orchestrates the complete data processing pipeline from raw measurements to
 * trained neural network models.
 */

/**
 * @brief Configuration options controlling calibration data processing.
 * 
 * ProcessCalibrationOptions contains all the parameters needed to configure
 * the calibration data processing pipeline. It includes experimental setup
 * parameters, robot configuration details, and processing options that control
 * how raw identification data is converted into calibration maps.
 */
struct ProcessCalibrationOptions {
    std::string data_path;       ///< Path to input data directory containing raw measurements
    int poses;                   ///< Number of pose samples in the identification experiment
    int axes;                    ///< Number of commanded axes during identification
    std::string robot_name = "test"; ///< Robot identifier/name for the experiment
    std::string file_format = "csv"; ///< Input file format ("csv", "npy", "npz")
    int num_joints = 6;          ///< Number of joints in the robot
    double min_freq = 1.0;       ///< Minimum excitation frequency in Hz
    double max_freq = 60.0;      ///< Maximum excitation frequency in Hz
    double freq_space = 0.5;     ///< Frequency spacing between excitation points in Hz
    double max_disp = M_PI / 36.0; ///< Maximum displacement amplitude in radians
    double dwell = 0.0;          ///< Dwell time between frequency sweeps in seconds
    double Ts = 0.004;           ///< Sampling period in seconds
    std::string sysid_type = "sine"; ///< Type of system identification ("sine", "chirp", etc.)
    std::string ctrl_config = "joint"; ///< Control configuration used ("joint", "cartesian", etc.)
    double max_acc = 2.0;        ///< Maximum acceleration limit in rad/s²
    double max_vel = 18.0;       ///< Maximum velocity limit in rad/s
    int sine_cycles = 5;         ///< Number of sine cycles per frequency excitation
    std::string sensor = "ToolAcc"; ///< Sensor used for measurements ("ToolAcc", "JointTorque", etc.)
    int start_pose = 0;          ///< Starting pose index for processing
    int max_map_size = 12;       ///< Maximum number of poses per calibration map
    bool saved_maps = false;     ///< If true, calibration maps are already stored on disk
};

/**
 * @brief Process raw calibration data using provided configuration options.
 * 
 * Main processing function that orchestrates the complete calibration data
 * pipeline. This function reads raw sine sweep data, performs frequency domain
 * analysis, extracts dynamics parameters, and generates calibration maps ready
 * for neural network training. It handles the entire workflow from raw data
 * to trained models.
 * 
 * @param opts Processing configuration options containing all necessary parameters.
 * @return Status code, 0 on success, non-zero on error.
 */
int processCalibrationData(const ProcessCalibrationOptions &opts);

/**
 * @brief Command-line interface for calibration data processing.
 * 
 * Provides a user-friendly command-line interface for processing calibration
 * data. This function parses command-line arguments, validates parameters,
 * and calls the main processing function with appropriate configuration.
 * It supports various command-line options for customizing the processing
 * behavior and provides help text for user guidance.
 * 
 * @param argc Argument count from main().
 * @param argv Argument vector from main().
 * @return Status code, 0 on success, non-zero on error or invalid arguments.
 */
int processCalibrationDataCLI(int argc, char **argv);