# Calibration Data Directory

This directory stores calibration data collected during robot system identification and calibration procedures.

## Overview

The `data/` directory contains raw and processed calibration data from robot system identification experiments. This data is used by the identification module to generate mathematical models of robot dynamics.

## Directory Structure

```
data/
├── README.md              # This file
├── test/                  # Test calibration data
│   ├── pose_0/           # Data from pose 0
│   ├── pose_1/           # Data from pose 1
│   └── pose_2/           # Data from pose 2
└── [robot_name]/         # Robot-specific data directories
    ├── calibration_001/  # Calibration session 001
    ├── calibration_002/  # Calibration session 002
    └── ...
```

## Data Organization

### Session Structure
Each calibration session contains:

- **Raw Data Files**: Sensor measurements in various formats
- **Configuration Files**: Calibration parameters and settings
- **Metadata**: Session information and timestamps
- **Results**: Processed calibration results

### File Naming Convention
- `pose_[N]_[sensor_type].[format]`: Data from pose N with specific sensor
- `config_[session_id].json`: Configuration for calibration session
- `metadata_[session_id].json`: Session metadata and timestamps
- `results_[session_id].[format]`: Processed calibration results

## Data Types

### Sensor Data
- **Joint Positions**: Encoder readings from robot joints
- **Joint Velocities**: Computed joint angular velocities
- **Joint Currents**: Motor current measurements
- **Accelerometer Data**: Tool acceleration measurements
- **IMU Data**: Inertial measurement unit readings
- **Timing Data**: Precise timestamps for all measurements

### Trajectory Data
- **Commanded Trajectories**: Input trajectories sent to robot
- **Actual Trajectories**: Measured robot responses
- **Trajectory Parameters**: Motion planning parameters
- **System Identification Parameters**: Identification experiment settings

### Calibration Results
- **Dynamic Parameters**: Identified mass, damping, and stiffness
- **Transfer Functions**: Frequency domain models
- **Calibration Maps**: Multi-dimensional parameter maps
- **Validation Data**: Model validation results

## Supported Formats

### Input Formats
- **CSV**: Comma-separated values with headers
- **NPZ**: NumPy compressed format
- **NPY**: NumPy binary format
- **JSON**: Configuration and metadata

### Output Formats
- **NPZ**: Processed calibration data
- **JSON**: Calibration results and parameters
- **CSV**: Tabular results for analysis
- **MAT**: MATLAB-compatible format

## Data Collection

### Automatic Collection
Data is automatically collected during calibration using:
- `src/calibration/autoCalibration.py`: Main calibration script
- Robot interface data recording
- Real-time sensor data acquisition

### Manual Collection
For manual data collection:
1. Ensure proper sensor calibration
2. Follow data collection protocols
3. Use consistent file naming
4. Include metadata and configuration

## Data Processing

### Preprocessing
- **Signal Filtering**: Remove noise and artifacts
- **Synchronization**: Align data from different sensors
- **Validation**: Check data quality and completeness
- **Normalization**: Standardize data formats

### Analysis
- **System Identification**: Extract dynamic parameters
- **Model Fitting**: Fit mathematical models to data
- **Validation**: Compare models with experimental data
- **Optimization**: Refine model parameters

## Usage

### Loading Data
```python
from src.util.Utility import load_data_file

# Load calibration data
data, headers = load_data_file("csv", "data/test/pose_0/joint_positions.csv")
```

### Processing Data
```python
# Process calibration data
python -m src.identification.python.process_calibration_data \
  data/test 3 3 \
  --format csv --name test
```

### Data Analysis
```python
import numpy as np
import matplotlib.pyplot as plt

# Load and analyze data
data = np.load("data/test/results.npz")
frequencies = data['frequencies']
responses = data['responses']

# Plot frequency response
plt.plot(frequencies, np.abs(responses))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()
```

## Data Management

### Backup Strategy
- **Regular Backups**: Automated backup of calibration data
- **Version Control**: Track changes to processed data
- **Archive Old Data**: Move old sessions to archive
- **Data Validation**: Verify data integrity

### Storage Requirements
- **Raw Data**: High storage requirements (GB per session)
- **Processed Data**: Compressed format (MB per session)
- **Metadata**: Minimal storage (KB per session)
- **Results**: Moderate storage (MB per session)

## Quality Control

### Data Validation
- **Completeness**: Check for missing data points
- **Consistency**: Verify data format and units
- **Accuracy**: Validate against known references
- **Timing**: Check synchronization between sensors

### Error Handling
- **Missing Files**: Graceful handling of missing data
- **Corrupted Data**: Detection and reporting of corruption
- **Format Errors**: Validation of file formats
- **Size Limits**: Check for reasonable data sizes

## Security

### Access Control
- **Read-only Access**: Protect calibration data from modification
- **Backup Protection**: Secure backup storage
- **Version Tracking**: Track all data modifications
- **Audit Trail**: Log all data access and changes

### Data Privacy
- **Robot Information**: Protect robot-specific data
- **Configuration Details**: Secure calibration parameters
- **Results Protection**: Protect proprietary calibration results