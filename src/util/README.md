# Utility Module

This module contains general utilities that are shared between files in the repository, providing common functionality for robot control, data processing, and mathematical operations.

## Overview

The utility module provides a collection of reusable components for robot dynamics, trajectory generation, data processing, and mathematical utilities. These utilities form the foundation for the calibration, identification, and control modules.

## Components

### Core Utilities

#### `Utility.py`
Main utility functions and data structures:

- **Data Structures**:
  - `MotionProfile`: Trajectory time and position data
  - `TrajParams`: Trajectory parameters (displacement, velocity, acceleration)
  - `SystemIdParams`: System identification parameters
  - `DataRecorder`: Data collection and storage
  - `StructuredCalibrationData`: Organized calibration data

- **Data Processing**:
  - `load_data_file()`: Load data from CSV, NPZ, or NPY formats
  - `store_recorder_data_in_csv()`: Save recorded data to CSV
  - `normalized_interpolation()`: Interpolate data with normalization
  - `get_polar_coordinates()`: Convert Cartesian to polar coordinates
  - `cartesian_to_polar()`: Coordinate system conversions

#### `Robot.py`
Base robot class and trajectory utilities:

- **`Robot`**: Abstract base class for robot implementations
- **`Trajectory`**: Trajectory generation and manipulation
- **Trajectory parameters and validation**

#### `RobotDynamics.py`
Robot dynamics and mathematical modeling:

- **`Dynamics`**: Robot dynamic model implementation
- **Kinematic calculations**: Forward/inverse kinematics
- **Dynamic modeling**: Mass, Coriolis, and gravity matrices
- **Jacobian computations**: Geometric and analytical Jacobians

#### `RobotInterface.py`
Abstract interface for robot communication:

- **`RobotInterface`**: Base class for robot control interfaces
- **Standardized robot communication protocols**
- **Error handling and safety features**

#### `Trajectory.py`
Advanced trajectory generation and manipulation:

- **Trajectory planning algorithms**
- **Smooth trajectory generation**
- **Velocity and acceleration profiling**
- **Multi-segment trajectory support**

### URDF Utilities (`urdf_to_dh/`)

Tools for converting URDF robot descriptions to Denavit-Hartenberg parameters:

- **`generate_dh.py`**: Main DH parameter generation
- **`geometry_helpers.py`**: Geometric transformation utilities
- **`kinematics_helpers.py`**: Kinematic calculation helpers
- **`urdf_helpers.py`**: URDF parsing and processing

## Usage

### Basic Data Processing

```python
from src.util.Utility import load_data_file, DataRecorder

# Load calibration data
data, headers = load_data_file("csv", "calibration_data.csv")

# Record robot data
recorder = DataRecorder()
recorder.inputJointPositions.append(joint_positions)
recorder.outputJointPositions.append(actual_positions)
```

### Trajectory Generation

```python
from src.util.Robot import Trajectory
from src.util.Utility import TrajParams

# Create trajectory parameters
params = TrajParams(
    configuration="joint",
    max_displacement=0.1,
    max_velocity=18.0,
    max_acceleration=2.0,
    sysid_type="sine",
    single_pt_run_time=4.0
)

# Generate trajectory
trajectory = Trajectory(params)
waypoints = trajectory.generate_waypoints(start_pos, end_pos)
```

### Robot Dynamics

```python
from src.util.RobotDynamics import Dynamics

# Create dynamics model
dynamics = Dynamics(urdf_path="robot.urdf")

# Compute dynamic matrices
M = dynamics.mass_matrix(q)  # Mass matrix
C = dynamics.coriolis_matrix(q, qd)  # Coriolis matrix
G = dynamics.gravity_vector(q)  # Gravity vector
```

### Coordinate Transformations

```python
from src.util.Utility import cartesian_to_polar, polar_to_cartesian

# Convert between coordinate systems
r, theta = cartesian_to_polar(x, y)
x, y = polar_to_cartesian(r, theta)
```

## Data Formats

### Supported File Formats
- **CSV**: Comma-separated values with headers
- **NPZ**: NumPy compressed format
- **NPY**: NumPy binary format

### Data Structures
- **Time series data**: Position, velocity, acceleration
- **Sensor data**: Joint positions, accelerations, currents
- **Trajectory data**: Waypoints, timing, parameters
- **Calibration data**: Structured multi-dimensional arrays

## Integration

This module is used by:
- **Calibration Module**: Data recording and processing
- **Identification Module**: Data loading and preprocessing
- **Control Module**: Trajectory generation and dynamics
- **Robot Module**: Base classes and interfaces

## Mathematical Features

### Coordinate Systems
- Cartesian to polar coordinate conversion
- Rotation matrix operations
- Homogeneous transformations

### Interpolation
- Linear interpolation
- Normalized interpolation
- Smooth trajectory interpolation

### Data Processing
- Signal filtering and smoothing
- Statistical analysis
- Data validation and cleaning

## Performance

- **Optimized algorithms** for real-time applications
- **Memory efficient** data structures
- **Vectorized operations** using NumPy
- **C++ extensions** for computationally intensive tasks

## Error Handling

- **Comprehensive validation** of input parameters
- **Graceful error recovery** for data loading
- **Type checking** and parameter validation
- **Descriptive error messages** for debugging