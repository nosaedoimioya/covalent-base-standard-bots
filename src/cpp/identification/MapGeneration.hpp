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
     * @brief Number of stored poses in the map.
     */
    int num_positions() const { return num_positions_; }

    /**
     * @brief Number of commanded axes represented in the map.
     */
    int axes_commanded() const { return axes_commanded_; }

    /**
     * @brief Number of joints used to generate the map.
     */
    int num_joints() const { return num_joints_; }

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
    const Eigen::MatrixXd& allWn()      const { return allWn_; }
    const Eigen::MatrixXd& allZeta()    const { return allZeta_; }
    const Eigen::MatrixXd& allWn2()     const { return allWn2_; }
    const Eigen::MatrixXd& allZeta2()   const { return allZeta2_; }
    const Eigen::MatrixXd& allInertia() const { return allInertia_; }
    const Eigen::MatrixXd& allV()       const { return allV_; }
    const Eigen::MatrixXd& allR()       const { return allR_; }
    
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

    /**
     * @brief Save the calibration map to disk as a .npz file.
     * @param filename Destination file name.
     */
    void save_npz(const std::string &filename) const;

    /**
     * @brief Load a calibration map (.npz file) from disk.
     * @param filename Source file name.
     * @return Loaded CalibrationMap instance.
     */
    static CalibrationMap load_npz(const std::string &filename);

    // --- direct writers (so SineSweepReader can fill values)
    Eigen::MatrixXd& mutable_allWn()      { return allWn_; }
    Eigen::MatrixXd& mutable_allZeta()    { return allZeta_; }
    Eigen::MatrixXd& mutable_allWn2()     { return allWn2_; }
    Eigen::MatrixXd& mutable_allZeta2()   { return allZeta2_; }
    Eigen::MatrixXd& mutable_allInertia() { return allInertia_; }
    Eigen::MatrixXd& mutable_allV()       { return allV_; }
    Eigen::MatrixXd& mutable_allR()       { return allR_; }

private:
    // Primary and (optional) second mode fields (shape: [num_positions_, axes_commanded_])
    Eigen::MatrixXd allWn_;
    Eigen::MatrixXd allZeta_;
    Eigen::MatrixXd allWn2_;
    Eigen::MatrixXd allZeta2_;

    // Trainer features
    Eigen::MatrixXd allInertia_;
    Eigen::MatrixXd allV_;   // radians
    Eigen::MatrixXd allR_;   // meters

    int num_positions_;
    int axes_commanded_;
    int num_joints_;

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