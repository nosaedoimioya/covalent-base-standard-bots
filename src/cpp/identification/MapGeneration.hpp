#pragma once

#include <Eigen/Dense>
#include <fftw3.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <complex>

namespace py = pybind11;

class CalibrationMap {
public:
    CalibrationMap(int numPositions, int axesCommanded, int numJoints);

    void generateCalibrationMap(py::array_t<double> robot_input,
                               py::array_t<double> robot_output,
                               double input_Ts, double output_Ts,
                               double max_freq_fit = 15.0,
                               bool gravity_comp = false,
                               int shift_store_position = 0);

    void save_map(const std::string &filename) const;
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