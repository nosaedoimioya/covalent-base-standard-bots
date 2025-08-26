# Calibration Module

This module contains code for calibrating robots to improve their accuracy and performance through system identification and parameter estimation.

## Overview

The calibration module provides automated tools for robot calibration using system identification techniques. It collects data from robot movements and uses this data to identify dynamic parameters and improve robot performance.

## Components

### `autoCalibration.py`
The main calibration script that performs automated robot calibration:

- **System Identification**: Runs sine sweep or broadband chirp (BCB) system identification
- **Trajectory Generation**: Creates optimized trajectories for parameter identification
- **Data Collection**: Records joint positions, currents, and accelerometer data
- **Real-time Control**: Provides joint or task space control during calibration

### Key Features

- **Multiple Identification Types**: Supports both sine sweep and broadband chirp identification
- **Configurable Parameters**: Adjustable displacement, velocity, and acceleration limits
- **Flexible Control Modes**: Joint space or task space control
- **Data Recording**: Comprehensive data collection for post-processing
- **Safety Features**: Built-in limits and safety checks

## Usage

### Basic Calibration

```bash
python src/calibration/autoCalibration.py <robot_ip> <local_ip>
```

### Advanced Options

```bash
python src/calibration/autoCalibration.py <robot_ip> <local_ip> \
  --name "my_robot" \
  --type "sine" \
  --freq 250 \
  --ctrl "joint" \
  --mdisp 0.1 \
  --mvel 18.0 \
  --macc 2.0 \
  --runtime 4.0 \
  --nv 8 \
  --nr 4
```

### Parameters

- `robot_ip`: IP address of the robot
- `local_ip`: Local IP address for control
- `--name`: Robot name identifier
- `--type`: System identification type (`sine` or `bcb`)
- `--freq`: Sampling frequency (20-250 Hz)
- `--ctrl`: Control mode (`joint` or `task`)
- `--mdisp`: Maximum displacement
- `--mvel`: Maximum velocity
- `--macc`: Maximum acceleration
- `--runtime`: Runtime per pose in seconds
- `--nv`: Number of angles to sweep
- `--nr`: Number of radii to sweep

## Data Storage

Calibration data is stored in the `data/` subdirectory:

- Raw sensor data (joint positions, accelerations, currents)
- Trajectory information
- System identification parameters
- Calibration results

## Integration

This module works with:
- **Identification Module**: Provides data for model generation
- **Control Module**: Uses calibrated parameters for vibration suppression
- **Robot Interface**: Communicates with physical or simulated robots

## Safety Notes

- Always ensure the robot workspace is clear before running calibration
- Start with conservative motion limits
- Monitor the robot during calibration
- Have an emergency stop available
- Test in simulation before running on physical hardware