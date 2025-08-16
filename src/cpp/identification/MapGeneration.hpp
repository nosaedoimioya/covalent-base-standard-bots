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
 */

/**
 * @brief Represents a calibration map estimated from frequency data.
 */
class CalibrationMap {
public:
    /**
     * @brief Construct a map container.
     * @param numPositions Number of stored positions.
     * @param axesCommanded Number of commanded axes.
     * @param numJoints Number of joints.
     */
    CalibrationMap(int numPositions, int axesCommanded, int numJoints);

    /**
     * @brief Generate a calibration map from robot data.
     *
     * @param robot_input   Input commands to the robot.
     * @param robot_output  Robot response measurements.
     * @param input_Ts      Sampling time of the input sequence.
     * @param output_Ts     Sampling time of the output sequence.
     * @param max_freq_fit  Maximum frequency to fit (Hz).
     * @param gravity_comp  Whether gravity compensation is applied.
     * @param shift_store_position Index shift for storing positions.
     */
    void generateCalibrationMap(py::array_t<double> robot_input,
                               py::array_t<double> robot_output,
                               double input_Ts, double output_Ts,
                               double max_freq_fit = 15.0,
                               bool gravity_comp = false,
                               int shift_store_position = 0);
    
    /**
     * @brief Save the calibration map to disk.
     * @param filename Destination file name.
     */
    void save_map(const std::string &filename) const;

    /**
     * @brief Load a calibration map from disk.
     * @param filename Source file name.
     * @return Loaded CalibrationMap instance.
     */
    static CalibrationMap load_map(const std::string &filename);

private:
    Eigen::MatrixXd sineSweepNumerators_;
    Eigen::MatrixXd sineSweepDenominators_;
    Eigen::MatrixXd allWn_;
    Eigen::MatrixXd allZeta_;

    // helper routines
    Eigen::VectorXcd computeFFT(const Eigen::VectorXd &data) const;
    Eigen::VectorXd butterworthFilter(const Eigen::VectorXd &data,
                                      double cutoff, double fs) const;
    Eigen::VectorXcd interpolateFFT(const Eigen::VectorXcd &data,
                                    const Eigen::VectorXd &freq_in,
                                    const Eigen::VectorXd &freq_out) const;
    void fitTransferFunction(const Eigen::VectorXcd &H,
                             const Eigen::VectorXd &w,
                             int nB, int nA,
                             Eigen::VectorXd &b,
                             Eigen::VectorXd &a) const;
};