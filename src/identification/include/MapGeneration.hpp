#pragma once

#include <complex>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <fftw3.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @file MapGeneration.hpp
 * @brief Facilities for generating calibration maps from sine sweep data.
 * 
 * This module provides the core functionality for processing raw sine sweep
 * identification data and extracting robot dynamics parameters. It includes
 * frequency domain analysis, transfer function fitting, and calibration map
 * generation utilities that form the foundation of the robot identification
 * pipeline.
 */

/**
 * @brief Represents a calibration map estimated from frequency domain data.
 * 
 * CalibrationMap is the central data structure for storing robot dynamics
 * parameters extracted from sine sweep identification experiments. It contains
 * both primary and secondary mode parameters (natural frequencies and damping
 * ratios) along with associated features used for training neural networks.
 * The class provides comprehensive I/O capabilities and integrates with the
 * FFTW library for efficient frequency domain processing.
 */
class CalibrationMap {
public:
    /**
     * @brief Construct a map container with specified dimensions.
     * 
     * Initializes a CalibrationMap with the given dimensions for storing
     * dynamics parameters across multiple poses and axes. The constructor
     * allocates memory for all parameter matrices and sets up the internal
     * data structures.
     * 
     * @param numPositions Number of stored positions/poses in the map.
     * @param axesCommanded Number of commanded axes during identification.
     * @param numJoints Number of joints in the robot.
     */
    CalibrationMap(int numPositions, int axesCommanded, int numJoints);

    /**
     * @brief Number of stored poses in the calibration map.
     * @return Number of position samples stored in the map.
     */
    int num_positions() const { return num_positions_; }

    /**
     * @brief Number of commanded axes represented in the map.
     * @return Number of axes that were actively commanded during identification.
     */
    int axes_commanded() const { return axes_commanded_; }

    /**
     * @brief Number of joints used to generate the map.
     * @return Number of joints in the robot configuration.
     */
    int num_joints() const { return num_joints_; }

    /**
     * @brief Generate a calibration map from robot input-output data.
     *
     * Processes raw robot identification data to extract dynamics parameters
     * using frequency domain analysis. The function performs FFT analysis,
     * transfer function fitting, and parameter extraction for each pose and
     * axis combination. It handles both single and dual-mode dynamics models.
     *
     * @param robot_input   Input commands to the robot (joint positions/velocities).
     * @param robot_output  Robot response measurements (accelerations/forces).
     * @param input_Ts      Sampling time of the input sequence in seconds.
     * @param output_Ts     Sampling time of the output sequence in seconds.
     * @param max_freq_fit  Maximum frequency to fit in Hz (default: 15.0).
     * @param gravity_comp  Whether gravity compensation is applied (default: false).
     * @param shift_store_position Index shift for storing positions (default: 0).
     */
    void generateCalibrationMap(py::array_t<double> robot_input,
                               py::array_t<double> robot_output,
                               double input_Ts, double output_Ts,
                               double max_freq_fit = 15.0,
                               bool gravity_comp = false,
                               int shift_store_position = 0);

    /**
     * @brief Generate calibration data from a single pose experiment.
     * 
     * Processes data from a single robot pose to extract dynamics parameters.
     * This function handles the complete pipeline from raw time series data
     * to extracted dynamics parameters, including signal processing, frequency
     * analysis, and parameter fitting.
     * 
     * @param q_cmd              Commanded joint positions [N x num_joints_].
     * @param tcp_acc            Tool center point accelerations [N x 3].
     * @param input_Ts           Input sampling time in seconds.
     * @param output_Ts          Output sampling time in seconds.
     * @param segments           Time segments for analysis as (start, end) pairs.
     * @param freq_cmd           Commanded frequencies for each segment.
     * @param robot_model        Python robot model for kinematics calculations.
     * @param V_angle            Joint angle in radians.
     * @param R_radius           Radius parameter in meters.
     * @param axis_commanded     Commanded axis index (1-based).
     * @param q0                 Reference joint configuration [num_joints_].
     * @param max_freq_fit       Maximum frequency to fit in Hz (default: 15.0).
     * @param gravity_comp       Whether gravity compensation is applied (default: false).
     * @param store_row          Row index for storing results (0-based).
     * @param store_axis         Axis index for storing results (0-based).
     */
    void generate_from_pose(
                            const Eigen::MatrixXd& q_cmd,              // [N x num_joints_]
                            const Eigen::MatrixXd& tcp_acc,            // [N x 3]
                            double input_Ts, double output_Ts,
                            const std::vector<std::pair<int,int>>& segments,
                            const std::vector<double>& freq_cmd,
                            py::object robot_model,
                            double V_angle, double R_radius,
                            int axis_commanded,                         // 1-based
                            const std::vector<double>& q0,              // size = num_joints_
                            double max_freq_fit = 15.0, 
                            bool gravity_comp = false,
                            int store_row = 0, int store_axis = 0               // 0-based
                        );
    
    // Accessors needed by training (exposed to Python as numpy arrays)
    /**
     * @brief Access primary natural frequencies.
     * @return Matrix of primary natural frequencies [num_positions_ x axes_commanded_].
     */
    const Eigen::MatrixXd& allWn()      const { return allWn_; }
    
    /**
     * @brief Access primary damping ratios.
     * @return Matrix of primary damping ratios [num_positions_ x axes_commanded_].
     */
    const Eigen::MatrixXd& allZeta()    const { return allZeta_; }
    
    /**
     * @brief Access secondary natural frequencies.
     * @return Matrix of secondary natural frequencies [num_positions_ x axes_commanded_].
     */
    const Eigen::MatrixXd& allWn2()     const { return allZeta2_; }
    
    /**
     * @brief Access secondary damping ratios.
     * @return Matrix of secondary damping ratios [num_positions_ x axes_commanded_].
     */
    const Eigen::MatrixXd& allZeta2()   const { return allZeta2_; }
    
    /**
     * @brief Access inertia parameters.
     * @return Matrix of inertia parameters [num_positions_ x axes_commanded_].
     */
    const Eigen::MatrixXd& allInertia() const { return allInertia_; }
    
    /**
     * @brief Access joint angle parameters.
     * @return Matrix of joint angles in radians [num_positions_ x axes_commanded_].
     */
    const Eigen::MatrixXd& allV()       const { return allV_; }
    
    /**
     * @brief Access radius parameters.
     * @return Matrix of radius parameters in meters [num_positions_ x axes_commanded_].
     */
    const Eigen::MatrixXd& allR()       const { return allR_; }
    
    /**
     * @brief Save the calibration map to disk in binary format.
     * 
     * Persists the calibration map data to a binary file for later loading.
     * The file format is optimized for fast I/O and compact storage.
     * 
     * @param filename Destination file name for the calibration map.
     */
    void save_map(const std::string &filename) const;

    /**
     * @brief Load a calibration map from disk.
     * 
     * Restores a CalibrationMap instance from a previously saved binary file.
     * The function automatically detects the file format and loads all stored
     * parameters and metadata.
     * 
     * @param filename Source file name containing the saved calibration map.
     * @return Loaded CalibrationMap instance with all parameters restored.
     */
    static CalibrationMap load_map(const std::string &filename);

    /**
     * @brief Save the calibration map to disk as a .npz file.
     * 
     * Exports the calibration map data in NumPy compressed archive format
     * for compatibility with Python tools and other analysis software.
     * 
     * @param filename Destination file name with .npz extension.
     */
    void save_npz(const std::string &filename) const;

    /**
     * @brief Load a calibration map (.npz file) from disk.
     * 
     * Imports calibration map data from a NumPy compressed archive file.
     * This function provides compatibility with Python-generated calibration maps.
     * 
     * @param filename Source file name with .npz extension.
     * @return Loaded CalibrationMap instance with all parameters restored.
     */
    static CalibrationMap load_npz(const std::string &filename);

    // --- direct writers (so SineSweepReader can fill values)
    /**
     * @brief Mutable access to primary natural frequencies.
     * @return Reference to matrix of primary natural frequencies.
     */
    Eigen::MatrixXd& mutable_allWn()      { return allWn_; }
    
    /**
     * @brief Mutable access to primary damping ratios.
     * @return Reference to matrix of primary damping ratios.
     */
    Eigen::MatrixXd& mutable_allZeta()    { return allZeta_; }
    
    /**
     * @brief Mutable access to secondary natural frequencies.
     * @return Reference to matrix of secondary natural frequencies.
     */
    Eigen::MatrixXd& mutable_allWn2()     { return allWn2_; }
    
    /**
     * @brief Mutable access to secondary damping ratios.
     * @return Reference to matrix of secondary damping ratios.
     */
    Eigen::MatrixXd& mutable_allZeta2()   { return allZeta2_; }
    
    /**
     * @brief Mutable access to inertia parameters.
     * @return Reference to matrix of inertia parameters.
     */
    Eigen::MatrixXd& mutable_allInertia() { return allInertia_; }
    
    /**
     * @brief Mutable access to joint angle parameters.
     * @return Reference to matrix of joint angles.
     */
    Eigen::MatrixXd& mutable_allV()       { return allV_; }
    
    /**
     * @brief Mutable access to radius parameters.
     * @return Reference to matrix of radius parameters.
     */
    Eigen::MatrixXd& mutable_allR()       { return allR_; }

private:
    // Primary and (optional) second mode fields (shape: [num_positions_, axes_commanded_])
    Eigen::MatrixXd allWn_;      ///< Primary natural frequencies in rad/s
    Eigen::MatrixXd allZeta_;    ///< Primary damping ratios (dimensionless)
    Eigen::MatrixXd allWn2_;     ///< Secondary natural frequencies in rad/s
    Eigen::MatrixXd allZeta2_;   ///< Secondary damping ratios (dimensionless)

    // Trainer features
    Eigen::MatrixXd allInertia_; ///< Inertia parameters in kg⋅m²
    Eigen::MatrixXd allV_;       ///< Joint angles in radians
    Eigen::MatrixXd allR_;       ///< Radius parameters in meters

    int num_positions_;          ///< Number of stored poses
    int axes_commanded_;         ///< Number of commanded axes
    int num_joints_;             ///< Number of robot joints

    // helper routines
    /**
     * @brief Compute FFT of input data using FFTW library.
     * 
     * Performs fast Fourier transform on the input data vector using the
     * FFTW library for optimal performance. Returns complex frequency domain data.
     * 
     * @param data Input time domain data vector.
     * @return Complex frequency domain representation.
     */
    Eigen::VectorXcd computeFFT(const Eigen::VectorXd &data) const;
    
    /**
     * @brief Apply Butterworth low-pass filter to input data.
     * 
     * Filters the input data using a digital Butterworth low-pass filter
     * to remove high-frequency noise and artifacts.
     * 
     * @param data Input data vector to filter.
     * @param cutoff Cutoff frequency in Hz.
     * @param fs Sampling frequency in Hz.
     * @return Filtered data vector.
     */
    Eigen::VectorXd butterworthFilter(const Eigen::VectorXd &data,
                                      double cutoff, double fs) const;
    
    /**
     * @brief Interpolate FFT data to different frequency points.
     * 
     * Resamples complex frequency domain data from input frequency points
     * to output frequency points using interpolation.
     * 
     * @param data Complex frequency domain data.
     * @param freq_in Input frequency points.
     * @param freq_out Output frequency points.
     * @return Interpolated complex frequency domain data.
     */
    Eigen::VectorXcd interpolateFFT(const Eigen::VectorXcd &data,
                                    const Eigen::VectorXd &freq_in,
                                    const Eigen::VectorXd &freq_out) const;
    
    /**
     * @brief Fit transfer function to frequency response data.
     * 
     * Performs system identification to fit a transfer function to the given
     * frequency response data. Uses least squares optimization to find optimal
     * numerator and denominator coefficients.
     * 
     * @param H Complex frequency response data.
     * @param w Frequency points in rad/s.
     * @param nB Number of numerator coefficients.
     * @param nA Number of denominator coefficients.
     * @param b Output numerator coefficients.
     * @param a Output denominator coefficients.
     */
    void fitTransferFunction(const Eigen::VectorXcd &H,
                             const Eigen::VectorXd &w,
                             int nB, int nA,
                             Eigen::VectorXd &b,
                             Eigen::VectorXd &a) const;
};