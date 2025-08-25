# Control Module - C++ Implementation with Python Bindings

This module contains the C++ implementation of the BaseShaper class for vibration suppression in mechanical systems, along with Python bindings for easy integration.

## Overview

The `BaseShaper` class implements Zero-Vibration-Derivative (ZVD) input shaping to suppress vibrations in mechanical systems. It can handle multiple vibration modes with varying natural frequencies and damping ratios.

## Files

- `BaseShaper.hpp` - Header file containing the class declaration
- `BaseShaper.cpp` - Implementation file containing all method definitions
- `BaseShaperBindings.cpp` - Python bindings using pybind11
- `test_base_shaper.cpp` - C++ test program to verify functionality
- `test_base_shaper_python.py` - Python test script using the bindings
- `setup_and_test.py` - Automated setup and test script
- `CMakeLists.txt` - Build configuration for the control module

## Key Features

### BaseShaper Class

The `BaseShaper` class provides the following functionality:

1. **Single Sample Shaping**: Process individual input samples with current dynamics parameters
2. **Trajectory Shaping**: Process entire trajectories with varying dynamics parameters
3. **ZVD Shaper Computation**: Generate Zero-Vibration-Derivative shapers for multiple vibration modes
4. **Online Processing**: Maintains internal buffer for efficient real-time processing

### Main Methods

- `BaseShaper(double Ts)` - Constructor with sampling time parameter
- `shape_sample(double x_i, const Eigen::MatrixXd& frf_params)` - Shape a single sample
- `shape_trajectory(const Eigen::VectorXd& x, const std::vector<Eigen::MatrixXd>& varying_params)` - Shape a complete trajectory
- `compute_zvd_shaper(const Eigen::MatrixXd& params_array)` - Compute ZVD shaper impulse response

## Dependencies

- **Eigen3**: Linear algebra library for matrix and vector operations
- **C++17**: Modern C++ features including structured bindings
- **pybind11**: Python bindings library
- **Python3**: Python interpreter and development headers

## Building

### C++ Only

To build the C++ library and test executable:

```bash
mkdir build
cd build
cmake ..
make
```

### With Python Bindings

To build with Python bindings:

```bash
mkdir build
cd build
cmake ..
make
```

This will create both the C++ library and the Python module `base_shaper`.

## Python Usage

After building, you can use the BaseShaper in Python:

```python
import numpy as np
import sys
import os

# Add the build directory to the Python path
sys.path.insert(0, os.path.join('build', 'src', 'cpp', 'control'))

import base_shaper

# Create BaseShaper with 0.001s sampling time
shaper = base_shaper.BaseShaper(0.001)

# Define dynamics parameters (natural frequency = 10 rad/s, damping = 0.1)
params = np.array([[10.0, 0.1]])

# Shape a single sample
input_sample = 1.0
shaped_sample = shaper.shape_sample(input_sample, params)

# Shape a trajectory
num_samples = 100
trajectory = np.zeros(num_samples)
trajectory[50:] = 1.0  # Step at sample 50

# Create varying parameters (same for all samples in this example)
varying_params = [params for _ in range(num_samples)]

shaped_trajectory = shaper.shape_trajectory(trajectory, varying_params)

# Compute ZVD shaper
impulse_response, filter_order = shaper.compute_zvd_shaper(params)
```

## Testing

### C++ Tests

Run the C++ test program:

```bash
./build/src/cpp/control/test_base_shaper
```

### Python Tests

Run the Python test script:

```bash
python src/control/test_base_shaper_python.py
```

### Automated Setup and Test

Use the setup script for a complete build and test:

```bash
python src/control/setup_and_test.py
```

## Differences from Python Version

The C++ implementation maintains the same mathematical algorithms as the Python version but includes:

1. **Strong Type Safety**: Compile-time type checking for all parameters
2. **Exception Handling**: Comprehensive error checking with descriptive error messages
3. **Memory Efficiency**: Optimized memory usage with Eigen's expression templates
4. **Performance**: Direct memory access and optimized numerical operations
5. **Python Integration**: Seamless integration with Python via pybind11 bindings

## Error Handling

The C++ implementation includes comprehensive error checking:

- Invalid sampling time (must be positive)
- Invalid dynamics parameters (natural frequency must be positive, damping ratio between 0 and 1)
- Dimension mismatches in input arrays
- Empty input arrays

All errors are reported via `std::invalid_argument` exceptions with descriptive messages, which are properly translated to Python exceptions when using the bindings.

## Installation

To install the Python module system-wide:

```bash
cd build
make install
```

The Python module will be installed to the system Python path and can be imported directly:

```python
import base_shaper
```
