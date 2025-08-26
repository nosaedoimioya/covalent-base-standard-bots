# Robots Module

This module contains robot-specific implementations and interfaces for different robot platforms, providing standardized access to robot control and data collection capabilities.

## Overview

The robots module provides concrete implementations of robot interfaces for various robot platforms. It includes the Standard Robots implementation and provides a framework for adding support for additional robot types.

## Components

### Robot Implementations

#### `Standard.py`
Main implementation for Standard Robots platform:

- **`StandardRobot`**: Complete robot interface for Standard Robots
- **System Identification**: Automated system identification capabilities
- **Real-time Control**: Joint and task space control
- **Data Collection**: Comprehensive sensor data recording
- **Safety Features**: Built-in safety limits and monitoring
- **Simulation Support**: Both live and simulated robot operation

#### `StandardRosManager.py`
ROS integration for Standard Robots:

- **ROS Control**: Integration with ROS control framework
- **Topic Management**: Standardized topic publishing/subscribing
- **Node Management**: ROS node lifecycle management
- **Message Handling**: ROS message type conversions

#### `TestRobot.py`
Test robot implementation for development and testing:

- **Simulated Robot**: Software-only robot for testing
- **Mock Data**: Simulated sensor data generation
- **Development Tool**: Safe environment for algorithm testing
- **Validation**: Robot behavior validation

### URDF Models (`urdf/`)

Robot description files:

- **`modelone.urdf`**: Standard robot URDF model
- **`test_robot.urdf`**: Test robot URDF model
- **Kinematic definitions**: Joint limits, link properties
- **Visual models**: Robot appearance for visualization

## Usage

### Basic Robot Control

```python
from src.robots.Standard import StandardRobot

# Initialize robot
robot = StandardRobot(robot_id="your_robot_id", api_token="your_token")

# Enable robot
robot.enable()

# Move to position
robot.move_to_joint_position([0, 1.57, 0, 0, 0, 0])

# Get current position
current_pos = robot.get_joint_positions()
```

### System Identification

```python
# Run system identification
robot.run_system_identification(
    sysid_type="sine",
    num_angles=8,
    num_radii=4,
    runtime=4.0
)

# Get recorded data
data = robot.recorder.get_data()
```

### ROS Integration

```python
from src.robots.StandardRosManager import StandardRosManager

# Initialize ROS manager
ros_manager = StandardRosManager(robot)

# Enable ROS control
ros_manager.enable_ros_control()

# Publish joint states
ros_manager.publish_joint_states()
```

## Robot Features

### Control Modes
- **Joint Space Control**: Direct joint position/velocity control
- **Task Space Control**: Cartesian position/orientation control
- **Trajectory Following**: Smooth trajectory execution
- **Real-time Control**: High-frequency control loops

### Data Collection
- **Joint Positions**: Current joint angles
- **Joint Velocities**: Joint angular velocities
- **Joint Currents**: Motor current measurements
- **Accelerometer Data**: Tool acceleration measurements
- **IMU Data**: Inertial measurement unit data
- **Timing Information**: Precise timing for all measurements

### Safety Features
- **Joint Limits**: Software-enforced joint limits
- **Velocity Limits**: Maximum velocity constraints
- **Acceleration Limits**: Maximum acceleration constraints
- **Emergency Stop**: Emergency stop functionality
- **Collision Detection**: Basic collision avoidance
- **Workspace Limits**: Cartesian workspace boundaries

## Configuration

### Robot Parameters
- **Robot ID**: Unique robot identifier
- **API Token**: Authentication token for robot access
- **URDF Path**: Path to robot description file
- **Control Frequency**: Control loop frequency (up to 250 Hz)
- **Safety Limits**: Joint and workspace limits

### Network Configuration
- **Robot IP**: Robot network address
- **Local IP**: Local network address for control
- **Port Configuration**: Network port settings
- **Timeout Settings**: Communication timeouts

## Integration

This module integrates with:
- **Calibration Module**: Provides robot interface for calibration
- **Control Module**: Uses robot data for vibration suppression
- **Identification Module**: Provides data for model generation
- **Utility Module**: Uses base classes and utilities

## Adding New Robots

To add support for a new robot platform:

1. **Create Robot Class**: Inherit from `src.util.Robot`
2. **Implement Interface**: Implement required methods
3. **Add URDF Model**: Create robot description file
4. **Test Implementation**: Validate with test suite
5. **Update Documentation**: Document robot-specific features

### Required Methods
```python
class NewRobot(Robot):
    def enable(self):
        """Enable robot control"""
        pass
    
    def disable(self):
        """Disable robot control"""
        pass
    
    def get_joint_positions(self):
        """Get current joint positions"""
        pass
    
    def move_to_joint_position(self, positions):
        """Move to joint positions"""
        pass
```

## Safety Guidelines

- **Always test in simulation** before running on physical hardware
- **Verify safety limits** are properly configured
- **Monitor robot during operation** for unexpected behavior
- **Have emergency stop available** at all times
- **Follow manufacturer guidelines** for robot operation
- **Regular maintenance** of robot hardware and software

## Troubleshooting

### Common Issues
- **Connection Problems**: Check network configuration
- **Authentication Errors**: Verify API token and permissions
- **Control Issues**: Check robot state and safety limits
- **Data Collection**: Verify sensor connections and calibration

### Debug Mode
Enable debug logging for detailed information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```