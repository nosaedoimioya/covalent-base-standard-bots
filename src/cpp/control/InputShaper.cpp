#include "InputShaper.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace control {

InputShaper::InputShaper(double Ts) : Ts_(Ts), M_(0) {
    if (Ts <= 0.0) {
        throw std::invalid_argument("Sampling time must be positive");
    }
}

double InputShaper::shape_sample(double x_i, const Eigen::MatrixXd& frf_params) {
    // Validate input parameters
    if (frf_params.cols() != 2) {
        throw std::invalid_argument("frf_params must have exactly 2 columns");
    }
    if (frf_params.rows() == 0) {
        throw std::invalid_argument("frf_params cannot be empty");
    }

    // Compute new shaper and its length
    auto [I, M_new] = compute_zvd_shaper(frf_params);

    // If first call or filter just grew, seed buffer with x_i
    if (buffer_.empty() || M_new != M_) {
        M_ = M_new;
        // prefill with the same initial value so no discontinuity
        buffer_.clear();
        buffer_.resize(M_ + 1, x_i);
    } else {
        // normal rolling: add newest, drop oldest automatically
        buffer_.push_front(x_i);
        if (buffer_.size() > M_ + 1) {
            buffer_.pop_back();
        }
    }

    // OSA convolution
    double x_shaped = 0.0;
    for (int i = 0; i <= M_; ++i) {
        x_shaped += I(i) * buffer_[i];
    }
    
    return x_shaped;
}

Eigen::VectorXd InputShaper::shape_trajectory(const Eigen::VectorXd& x, 
                                             const std::vector<Eigen::MatrixXd>& varying_params) {
    // Validate input dimensions
    if (x.size() == 0) {
        throw std::invalid_argument("Input trajectory cannot be empty");
    }
    
    if (varying_params.size() != static_cast<size_t>(x.size())) {
        throw std::invalid_argument("varying_params must have the same number of elements as x");
    }
    
    // Check if varying_params has exactly two columns
    if (varying_params[0].cols() != 2) {
        throw std::invalid_argument("varying_params must have 2 columns");
    }

    Eigen::VectorXd x_shaped(x.size());
    for (int i = 0; i < x.size(); ++i) {
        x_shaped(i) = shape_sample(x(i), varying_params[i]);
    }
    return x_shaped;
}

std::pair<Eigen::VectorXd, int> InputShaper::compute_zvd_shaper(const Eigen::MatrixXd& params_array) {
    // Validate input parameters
    if (params_array.cols() != 2) {
        throw std::invalid_argument("params_array must have exactly 2 columns");
    }
    if (params_array.rows() == 0) {
        throw std::invalid_argument("params_array cannot be empty");
    }

    // Initialize I and M with the first mode in params_array
    Eigen::VectorXd I;
    int M = 0;
    
    auto [I_single, d] = compute_single_mode(params_array(0, 0), params_array(0, 1));
    I = I_single;
    M = I.size() - 1;

    // Convolve the remaining modes
    for (int i = 1; i < params_array.rows(); ++i) {
        auto [I_new, d_new] = compute_single_mode(params_array(i, 0), params_array(i, 1));
        
        // Perform convolution manually since Eigen doesn't have a direct convolution function
        int new_size = I.size() + I_new.size() - 1;
        Eigen::VectorXd I_conv(new_size);
        I_conv.setZero();
        
        for (int j = 0; j < I.size(); ++j) {
            for (int k = 0; k < I_new.size(); ++k) {
                I_conv(j + k) += I(j) * I_new(k);
            }
        }
        
        I = I_conv;
        M = I.size() - 1;
    }

    return {I, M};
}

std::pair<Eigen::VectorXd, int> InputShaper::compute_single_mode(double wn, double zeta) {
    // Validate input parameters
    if (wn <= 0.0) {
        throw std::invalid_argument("Natural frequency must be positive");
    }
    if (zeta <= 0.0 || zeta >= 1.0) {
        throw std::invalid_argument("Damping ratio must be between 0 and 1");
    }

    double wd = wn * std::sqrt(1.0 - zeta * zeta);
    double k = std::exp(-zeta * M_PI / std::sqrt(1.0 - zeta * zeta));
    double Td = 2.0 * M_PI / wd;  // half-period of damped oscillation
    int d = static_cast<int>(std::round(0.5 * Td / Ts_));  // samples

    double norm = 1.0 + 2.0 * k + k * k;
    Eigen::VectorXd impulse(2 * d + 1);
    impulse.setZero();
    
    impulse(0) = 1.0 / norm;
    impulse(d) = 2.0 * k / norm;
    impulse(2 * d) = k * k / norm;

    return {impulse, d};
}

} // namespace control
