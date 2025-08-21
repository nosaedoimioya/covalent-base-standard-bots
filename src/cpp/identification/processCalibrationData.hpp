#pragma once

#include <cmath>
#include <string>

/**
 * @file ProcessCalibrationData.hpp
 * @brief Interfaces for converting raw identification data into calibration maps.
 */

/**
 * @brief Options controlling calibration data processing.
 */
struct ProcessCalibrationOptions {
    std::string data_path;       ///< Path to input data directory
    int poses;                   ///< Number of pose samples
    int axes;                    ///< Number of commanded axes
    std::string robot_name = "test"; ///< Robot identifier
    std::string file_format = "csv"; ///< Input file format
    int num_joints = 6;          ///< Number of joints in the robot
    double min_freq = 1.0;       ///< Minimum excitation frequency
    double max_freq = 60.0;      ///< Maximum excitation frequency
    double freq_space = 0.5;     ///< Frequency spacing
    double max_disp = M_PI / 36.0; ///< Maximum displacement
    double dwell = 0.0;          ///< Dwell time between sweeps
    double Ts = 0.004;           ///< Sampling period
    std::string sysid_type = "sine"; ///< Type of system identification
    std::string ctrl_config = "joint"; ///< Control configuration used
    double max_acc = 2.0;        ///< Maximum acceleration
    double max_vel = 18.0;       ///< Maximum velocity
    int sine_cycles = 5;         ///< Number of sine cycles per excitation
    std::string sensor = "ToolAcc"; ///< Sensor used for measurements
    int start_pose = 0;          ///< Starting pose index
    int max_map_size = 12;       ///< Maximum number of calibration maps
    bool saved_maps = false;     ///< If true, maps are already stored on disk
};

/**
 * @brief Process raw calibration data using provided options.
 * @param opts Processing options.
 * @return Status code, 0 on success.
 */
int processCalibrationData(const ProcessCalibrationOptions &opts);

/**
 * @brief Command-line interface for calibration processing.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Status code, 0 on success.
 */
int processCalibrationDataCLI(int argc, char **argv);