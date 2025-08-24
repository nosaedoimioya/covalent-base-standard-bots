# Control Module - C++ Implementation

This module contains the C++ implementation of the InputShaper class for vibration suppression in mechanical systems.

## Overview

The `InputShaper` class implements Zero-Vibration-Derivative (ZVD) input shaping to suppress vibrations in mechanical systems. It can handle multiple vibration modes with varying natural frequencies and damping ratios.

## Files

- `InputShaper.hpp` - Header file containing the class declaration
- `InputShaper.cpp` - Implementation file containing all method definitions
- `test_input_shaper.cpp` - Simple test program to verify functionality
- `CMakeLists.txt` - Build configuration for the control module

## Key Features

### InputShaper Class

The `InputShaper` class provides the following functionality:

1. **Single Sample Shaping**: Process individual input samples with current dynamics parameters
2. **Trajectory Shaping**: Process entire trajectories with varying dynamics parameters
3. **ZVD Shaper Computation**: Generate Zero-Vibration-Derivative shapers for multiple vibration modes
4. **Online Processing**: Maintains internal buffer for efficient real-time processing

### Main Methods

- `InputShaper(double Ts)` - Constructor with sampling time parameter
- `shape_sample(double x_i, const Eigen::MatrixXd& frf_params)` - Shape a single sample
- `shape_trajectory(const Eigen::VectorXd& x, const std::vector<Eigen::MatrixXd>& varying_params)` - Shape a complete trajectory
- `compute_zvd_shaper(const Eigen::MatrixXd& params_array)` - Compute ZVD shaper impulse response

## Dependencies

- **Eigen3**: Linear algebra library for matrix and vector operations
- **C++17**: Modern C++ features including structured bindings

## Building

To build the control module:

```bash
mkdir build
cd build
cmake ..
make
```

## Usage Example

```cpp
#include "InputShaper.hpp"
#include <Eigen/Dense>

using namespace control;

int main() {
    // Create InputShaper with 0.001s sampling time
    InputShaper shaper(0.001);
    
    // Define dynamics parameters (natural frequency = 10 rad/s, damping = 0.1)
    Eigen::MatrixXd params(1, 2);
    params << 10.0, 0.1;
    
    // Shape a single sample
    double input_sample = 1.0;
    double shaped_sample = shaper.shape_sample(input_sample, params);
    
    // Shape a trajectory
    int num_samples = 100;
    Eigen::VectorXd trajectory(num_samples);
    std::vector<Eigen::MatrixXd> varying_params(num_samples);
    
    // Create step trajectory
    for (int i = 0; i < num_samples; ++i) {
        trajectory(i) = (i < 50) ? 0.0 : 1.0;
        varying_params[i] = params;
    }
    
    Eigen::VectorXd shaped_trajectory = shaper.shape_trajectory(trajectory, varying_params);
    
    return 0;
}
```

## Testing

Run the test program to verify the implementation:

```bash
./test_input_shaper
```

## Differences from Python Version

The C++ implementation maintains the same mathematical algorithms as the Python version but includes:

1. **Strong Type Safety**: Compile-time type checking for all parameters
2. **Exception Handling**: Comprehensive error checking with descriptive error messages
3. **Memory Efficiency**: Optimized memory usage with Eigen's expression templates
4. **Performance**: Direct memory access and optimized numerical operations

## Error Handling

The C++ implementation includes comprehensive error checking:

- Invalid sampling time (must be positive)
- Invalid dynamics parameters (natural frequency must be positive, damping ratio between 0 and 1)
- Dimension mismatches in input arrays
- Empty input arrays

All errors are reported via `std::invalid_argument` exceptions with descriptive messages.
