#include "BaseShaper.hpp"
#include <iostream>
#include <iomanip>

using namespace control;

int main() {
    try {
        // Test basic functionality
        std::cout << "Testing BaseShaper C++ implementation..." << std::endl;
        
        // Create BaseShaper with 0.001s sampling time
        BaseShaper shaper(0.001);
        
        // Test single mode parameters (natural frequency = 10 rad/s, damping = 0.1)
        Eigen::MatrixXd params(1, 2);
        params << 10.0, 0.1;
        
        // Test single sample shaping
        double input_sample = 1.0;
        double shaped_sample = shaper.shape_sample(input_sample, params);
        
        std::cout << "Input sample: " << input_sample << std::endl;
        std::cout << "Shaped sample: " << shaped_sample << std::endl;
        
        // Test trajectory shaping
        int num_samples = 100;
        Eigen::VectorXd trajectory(num_samples);
        std::vector<Eigen::MatrixXd> varying_params(num_samples);
        
        // Create a simple step trajectory
        for (int i = 0; i < num_samples; ++i) {
            trajectory(i) = (i < 50) ? 0.0 : 1.0;  // Step at sample 50
            varying_params[i] = params;  // Same parameters for all samples
        }
        
        Eigen::VectorXd shaped_trajectory = shaper.shape_trajectory(trajectory, varying_params);
        
        std::cout << "Trajectory shaping completed successfully!" << std::endl;
        std::cout << "Original trajectory size: " << trajectory.size() << std::endl;
        std::cout << "Shaped trajectory size: " << shaped_trajectory.size() << std::endl;
        
        // Test ZVD shaper computation
        auto [impulse_response, filter_order] = shaper.compute_zvd_shaper(params);
        
        std::cout << "ZVD shaper computed successfully!" << std::endl;
        std::cout << "Impulse response length: " << impulse_response.size() << std::endl;
        std::cout << "Filter order: " << filter_order << std::endl;
        
        std::cout << "All tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
