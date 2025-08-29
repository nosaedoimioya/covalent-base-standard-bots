# Standard Bots Dynamic Calibration Guide

## Overview
Standard Bots robots can be dynamically calibrated to suppress vibration and improve positional accuracy. This repository provides all the components needed to collect calibration data, generate dynamic models, and integrate them into real‑time control.

The code base is organized into modules that mirror the calibration workflow:

| Module | Purpose |
| --- | --- |
| `src/calibration` | Run robot trajectories and record data |
| `src/identification` | Process recorded data to build dynamic models |
| `src/control` | Real‑time vibration suppression via Covalent Base |
| `src/robots` | Robot‑specific interfaces (Standard Bots implementation included) |
| `src/util` | Shared utilities for dynamics, trajectory generation, and data handling |

The guide below explains system requirements, repository layout, and step‑by‑step instructions for running a calibration with a Standard Bots robot.

## 1. System Setup
### Hardware
- Standard Bots robot with network connectivity
- Workstation PC running Ubuntu 20.04/22.04
- Wired LAN connection between PC and robot (≤1 ms round‑trip latency recommended)
- Tool‑center‑point (TCP) accelerometer for vibration measurements

### Software
- Python 3.10+
- Standard Bots SDK:
  ```bash
  pip install standardbots
  ```
- Project dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Build tools for C++ modules if using the identification/control extensions

## 2. Repository File Structure
```
├── CMakeLists.txt
├── README.md
├── docs/
│   └── StandardBotsCalibrationGuide.md
└── src/
    ├── calibration/
    │   ├── autoCalibration.py      # main calibration entry point
    │   ├── data/                   # collected run data
    │   └── models/                 # generated calibration maps
    ├── control/                    # C++ BaseShaper & bindings
    ├── identification/             # data processing and model fitting
    ├── robots/
    │   ├── Standard.py             # Standard Bots interface
    │   └── urdf/                   # robot URDF files
    └── util/                       # shared classes and helpers
```

## 3. Calibration Overview
Dynamic calibration measures how each joint of the robot responds to motion commands. The process has two stages:

1. **Data Collection** – `autoCalibration.py` moves the robot through a grid of poses and excites each axis with a sine‑sweep or bang‑coast‑bang trajectory. Joint positions, currents and TCP acceleration are logged at each time step.
2. **Model Generation** – tools in `src/identification` convert the raw logs into frequency‑response functions (FRFs) and train neural‑network models that predict joint dynamics over the workspace. These models feed the `src/control` shaper to cancel vibration in real time.

## 4. Getting Started
1. Ensure the robot is powered and the Standard Bots API is reachable.
2. From the repository root run:
   ```bash
   python src/calibration/autoCalibration.py <robot_ip> <api_token> \
       --name standard --type sine --freq 250 --ctrl joint
   ```
   Key options:
   - `--type` `bcb` or `sine`
   - `--freq` sampling frequency (Hz)
   - `--ctrl` `joint` or `task`
   - `--mdisp`, `--mvel`, `--macc` motion limits
   - `--nv`, `--nr` number of workspace angles and radii
3. Collected CSV files appear under `src/calibration/data/<date>/`, where `<date>` is in the yyyy-m-d format.

### Processing Data
After a run:
```bash
python -m src.identification.python.process_calibration_data \
    src/calibration/data/<date> <poses> <axes> \
    --format csv --name standard --numjoints 6 \
    --minfreq 1 --maxfreq 20 --timestep 0.004 \
    --ctrl joint --type sine
```
This generates calibration maps inside `src/calibration/models/`. See the `src/identification/python/process_calibration_data.py` for all options. If you used any of the options during the calibration step, ensure to include the identical options for data processing. For example, if you set --macc 2 during calibration, also set --macc 2 when processing the data.

## 5. New Robot Integration
To support a different robot (e.g., R03):
1. Create a subclass of `Robot` in `src/robots/` implementing the required interface methods (`move_home`, `command_robot`, etc.). Most of the code in the Standard.py implementation should remain the same.
2. Add an entry in `RobotInterface._create_robot` so `--name` resolves to the new class.
3. Provide an accurate URDF under `src/robots/urdf/`.
4. Follow the skeleton in `src/robots/README.md` for detailed guidance.

## 6. Running & Collecting Data
- Keep the workspace clear and have an emergency stop available.
- Start with conservative motion limits and gradually increase.
- Monitor log output for dropped packets or limit violations.
- Each pose produces `robotData_motion_pose<i>_axis<j>.csv` and `robotData_static_pose<i>_axis<j>.csv` files.

## 7. Control

The control module provides real-time vibration suppression using the Covalent Base input shaping technology. It implements Zero-Vibration-Derivative (ZVD) input shaping that can suppress multiple vibration modes simultaneously.

### Overview

The control system consists of:
- **BaseShaper**: C++ implementation with Python bindings for high-performance vibration suppression
- **Dynamic Parameter Loading**: Integration with identification results to load workspace-dependent dynamics
- **Real-time Processing**: Online trajectory shaping for live robot control
- **Multi-mode Suppression**: Handles multiple vibration modes per joint

### Building the Control Module

1. **Build C++ Extensions**:
   ```bash
   cmake -S . -B build
   cmake --build build
   ```

2. **Verify Installation**:
   ```bash
   # Test C++ implementation
   ./build/src/control/lib/test_base_shaper
   
   # Test Python bindings
   python src/control/python/test_base_shaper_python.py
   ```

### Integration with Standard Bots Robots

The control module is already integrated into the Standard Bots robot interface. The `StandardRobot` class includes:

- **Automatic Shaping**: Built-in trajectory shaping for system identification
- **Dynamic Parameter Loading**: Loads FRF parameters from identification results
- **Real-time Processing**: Online shaping during robot control

#### Key Integration Points

1. **FRF Parameter Loading**:
   ```python
   from src.identification.lib.MapFitter import ModelLoader
   
   # Load axis model
   model_loader = ModelLoader("path/to/axis_model")
   frf_params = model_loader.get_frf_parameters(joint_positions)
   ```

2. **Trajectory Shaping**:
   ```python
   from control.python.BaseShaper import BaseShaper
   
   # Create shaper with robot sampling time
   shaper = BaseShaper(Ts=0.004)  # 250 Hz sampling
   
   # Shape trajectory with varying dynamics
   shaped_trajectory = shaper.shape_trajectory(
       x=unshaped_trajectory,
       varying_params=frf_params
   )
   ```

### Using Vibration Suppression

#### 1. Basic Usage

```python
from src.robots.Standard import StandardRobot
from control.python.BaseShaper import BaseShaper

# Initialize robot with vibration suppression
robot = StandardRobot(robot_id="your_robot_id", api_token="your_token")

# The robot automatically uses BaseShaper for system identification
# and can be configured for general motion with vibration suppression
```

#### 2. Custom Trajectory Shaping

```python
import numpy as np
from src.identification.lib.MapFitter import ModelLoader

# Load your model results
model_loader = ModelLoader("src/calibration/models/")

# Create trajectory to shape
target_positions = np.array([0, np.pi/2, 0, 0, 0, 0])  # Home position
trajectory = generate_trajectory_to_target(target_positions)

# Get FRF parameters for current robot state
current_joints = robot.get_joint_positions()
frf_params = model_loader.get_frf_parameters(current_joints)

# Shape the trajectory
shaper = BaseShaper(Ts=0.004)
shaped_trajectory = shaper.shape_trajectory(
    x=trajectory,
    varying_params=frf_params
)

# Execute shaped trajectory
robot.execute_trajectory(shaped_trajectory)
```

### Configuration Options

#### Sampling Time
- **Default**: 0.004s (250 Hz) for Standard Bots robots
- **Range**: 0.001s to 0.01s (1000 Hz to 100 Hz)
- **Recommendation**: Match robot control frequency

#### Vibration Modes
- **Single Mode**: Basic vibration suppression
- **Multi-mode**: Suppress multiple frequencies per joint
- **Workspace-dependent**: Parameters vary with robot configuration

### Advanced Features

#### 1. Multi-joint Coordination
```python
# Shape multiple joints simultaneously
joint_trajectories = [traj1, traj2, traj3, traj4, traj5, traj6]
shaped_trajectories = []

for i, trajectory in enumerate(joint_trajectories):
    shaper = BaseShaper(Ts=0.004)
    frf_params = model_loader.get_frf_parameters_for_joint(i, robot_state)
    shaped = shaper.shape_trajectory(trajectory, frf_params)
    shaped_trajectories.append(shaped)
```

#### 2. Adaptive Shaping
```python
# Update parameters based on robot state
def adaptive_shaping(robot, target_position):
    current_state = robot.get_joint_positions()
    frf_params = model_loader.get_frf_parameters(current_state)
    
    shaper = BaseShaper(Ts=0.004)
    trajectory = generate_trajectory(current_state, target_position)
    
    return shaper.shape_trajectory(trajectory, frf_params)
```

#### 3. Performance Monitoring
```python
# Monitor vibration suppression effectiveness
def monitor_vibration(robot, shaped_trajectory):
    # Execute shaped trajectory
    robot.execute_trajectory(shaped_trajectory)
    
    # Measure actual vibration
    accelerometer_data = robot.get_tcp_acceleration()
    
    # Compare with expected suppression
    vibration_reduction = calculate_vibration_reduction(accelerometer_data)
    return vibration_reduction
```

### Troubleshooting

#### Common Issues

1. **Build Errors**:
   ```bash
   # Ensure all dependencies are installed
   sudo apt-get install libeigen3-dev
   pip install pybind11
   
   # Rebuild from scratch
   rm -rf build
   cmake -S . -B build
   cmake --build build
   ```

2. **Import Errors**:
   ```python
   # Check Python path includes build directory
   import sys
   sys.path.append('build/src/control/lib')
   
   # Verify module is built
   import base_shaper
   ```

3. **Performance Issues**:
   - Reduce sampling frequency for better real-time performance
   - Use single-mode shaping for simpler systems
   - Optimize trajectory generation for smoother motion

#### Performance Optimization

1. **Memory Management**:
   ```python
   # Pre-allocate shapers for each joint
   shapers = [BaseShaper(Ts=0.004) for _ in range(6)]
   
   # Reuse shapers instead of creating new ones
   for i, shaper in enumerate(shapers):
       shaped_traj = shaper.shape_trajectory(trajectories[i], frf_params[i])
   ```

2. **Parameter Caching**:
   ```python
   # Cache FRF parameters for common configurations
   parameter_cache = {}
   
   def get_cached_frf_params(joint_config):
       config_key = tuple(np.round(joint_config, 2))
       if config_key not in parameter_cache:
           parameter_cache[config_key] = model_loader.get_frf_parameters(joint_config)
       return parameter_cache[config_key]
   ```

### Safety Considerations

1. **Emergency Stop**: Always have emergency stop available
2. **Workspace Limits**: Verify shaped trajectories respect robot limits
3. **Parameter Validation**: Ensure FRF parameters are within expected ranges
4. **Gradual Testing**: Start with conservative shaping parameters
5. **Monitoring**: Continuously monitor robot behavior during shaped motion

### Integration Checklist

- [ ] Control module built successfully
- [ ] Python bindings working
- [ ] Calibration maps loaded correctly
- [ ] FRF parameters validated
- [ ] Sampling time configured
- [ ] Emergency stop tested
- [ ] Performance validated
- [ ] Safety limits verified

## 8. Support
For questions about Standard Bots integration or calibration workflow, contact support@reforgerobotics.com.
