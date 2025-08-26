#pragma once

#include <vector>
#include <deque>
#include <Eigen/Dense>
#include <stdexcept>

/**
 * @file BaseShaper.hpp
 * @brief C++ implementation of input signal shaping for vibration suppression.
 * 
 * This module provides the BaseShaper class for shaping input signals to suppress
 * vibrations in mechanical systems. It implements Zero-Vibration-Derivative (ZVD)
 * shapers that can handle multiple vibration modes with varying dynamics parameters.
 * The class maintains an internal buffer for online signal processing and supports
 * both single-sample and trajectory-based shaping.
 */

namespace control {

/**
 * @brief Base signal shaper for vibration suppression using ZVD filters.
 * 
 * BaseShaper implements Zero-Vibration-Derivative (ZVD) input shaping to suppress
 * vibrations in mechanical systems. It can handle multiple vibration modes with
 * varying natural frequencies and damping ratios. The class maintains an internal
 * buffer for online processing and supports both single-sample and batch trajectory
 * shaping operations.
 */
class BaseShaper {
public:
    /**
     * @brief Construct an BaseShaper with specified sampling time.
     * 
     * Initializes the base shaper with the given sampling time. The internal
     * buffer is initialized empty and will be populated during the first shaping
     * operation.
     * 
     * @param Ts Sampling time in seconds.
     */
        explicit BaseShaper(double Ts);

    /**
     * @brief Shape a single input sample using current dynamics parameters.
     * 
     * Processes a single input sample using the specified dynamics parameters
     * (natural frequency and damping ratio). The function maintains an internal
     * buffer of past samples and recomputes the impulse response vector for each
     * call to handle varying dynamics. Uses Overlap-Save Algorithm (OSA) convolution
     * for efficient processing.
     * 
     * @param x_i Input sample to shape.
     * @param frf_params Matrix of dynamics parameters [num_modes x 2] where each row
     *                   contains [natural_frequency, damping_ratio].
     * @return Shaped output sample.
     */
    double shape_sample(double x_i, const Eigen::MatrixXd& frf_params);

    /**
     * @brief Shape a complete trajectory using varying dynamics parameters.
     * 
     * Processes an entire trajectory using dynamics parameters that may vary
     * at each time step. This function applies the base shaper to each sample
     * in the trajectory using the corresponding dynamics parameters.
     * 
     * @param x Input trajectory vector [num_samples x 1].
     * @param varying_params Vector of dynamics parameter matrices, one for each
     *                       time step. Each matrix has shape [num_modes x 2].
     * @return Shaped trajectory vector [num_samples x 1].
     * @throws std::invalid_argument if input dimensions don't match or parameters are invalid.
     */
    Eigen::VectorXd shape_trajectory(const Eigen::VectorXd& x, 
                                    const std::vector<Eigen::MatrixXd>& varying_params);

    /**
     * @brief Compute ZVD shaper impulse response for given dynamics parameters.
     * 
     * Creates a Zero-Vibration-Derivative (ZVD) shaper by convolving multiple
     * single-mode ZV filters. The resulting impulse response can be used for
     * vibration suppression of systems with multiple vibration modes.
     * 
     * @param params_array Matrix of dynamics parameters [num_modes x 2] where each row
     *                     contains [natural_frequency, damping_ratio].
     * @return Pair containing the impulse response vector and filter order (max delay index).
     * @throws std::invalid_argument if parameters array is invalid.
     */
    std::pair<Eigen::VectorXd, int> compute_zvd_shaper(const Eigen::MatrixXd& params_array);

private:
    double Ts_;                    ///< Sampling time in seconds
    std::deque<double> buffer_;    ///< Buffer for past samples (newest at front)
    int M_;                        ///< Current filter length - 1

    /**
     * @brief Compute single-mode ZVD shaper impulse response.
     * 
     * Builds a single-mode Zero-Vibration-Derivative (ZVD) shaper impulse train
     * with the transfer function:
     * 
     * where k is the damping factor and d is the delay index.
     * 
     * @param wn Natural frequency in rad/s.
     * @param zeta Damping ratio (0 < zeta < 1).
     * @return Pair containing the impulse response vector and delay index.
     */
    std::pair<Eigen::VectorXd, int> compute_single_mode(double wn, double zeta);
};

} // namespace control
