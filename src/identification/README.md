# Identification Module

This module contains code for generating robot models after calibration, providing system identification and parameter estimation capabilities.

## Overview

The identification module processes calibration data to generate mathematical models of robot dynamics. It includes both Python interfaces and high-performance C++ implementations for efficient data processing and model generation.

## Components

### Python Interface (`python/`)
- **`process_calibration_data.py`**: Main CLI for processing calibration data
- **`fine_tune_model.py`**: Fine-tuning of generated models

### C++ Extensions (`lib/`)
High-performance compiled extensions for data processing:

- **`MapGeneration`**: Generates calibration maps from raw data
- **`MapFitter`**: Fits mathematical models to calibration data
- **`SineSweepReader`**: Processes sine sweep identification data
- **`FineTuneModelGen`**: Generates fine-tuned dynamic models
- **`ProcessCalibrationData`**: Main data processing pipeline

### Core Libraries
- **`ident_core`**: Core identification algorithms
- **`mapfitter_core`**: Model fitting algorithms
- **`sinesweep_core`**: Sine sweep data processing
- **`LegacyMapLoader`**: Legacy data format support
- **`NPZMapLoader`**: NumPy data format support

## Usage

### Processing Calibration Data

```bash
python -m src.identification.python.process_calibration_data \
  <data_path> <poses> <axes> \
  --format csv --name test --numjoints 6 \
  --minfreq 1 --maxfreq 20.5 \
  --timestep 0.005 --dwell 1.0 --macc 2
```

### Parameters

- `data_path`: Path to calibration data folder
- `poses`: Number of poses robot visited
- `axes`: Number of commanded axes per pose
- `--format`: Data format (`csv`, `npz`, `npy`)
- `--name`: Robot name identifier
- `--numjoints`: Number of robot joints
- `--minfreq`/`--maxfreq`: Frequency range for identification
- `--timestep`: Sampling time step
- `--dwell`: Dwell time between movements
- `--macc`: Maximum acceleration

### Python API

```python
from src.identification import processCalibrationData, SineSweepReader

# Process calibration data
processCalibrationData(
    data_path="path/to/data",
    poses=3,
    axes=3,
    robot_name="test",
    file_format="csv"
)

# Use sine sweep reader
reader = SineSweepReader()
# ... process data
```

## Data Formats

### Supported Input Formats
- **CSV**: Comma-separated values with headers
- **NPZ**: NumPy compressed format
- **NPY**: NumPy binary format

### Output Formats
- Calibration maps (JSON/NumPy)
- Dynamic model parameters
- Frequency response data
- System identification results

## Model Types

### Sine Sweep Identification
- Frequency domain analysis
- Transfer function estimation
- Resonance frequency identification
- Damping ratio estimation

### Broadband Chirp (BCB)
- Time domain analysis
- Impulse response estimation
- System order identification
- Parameter uncertainty quantification

## Integration

This module integrates with:
- **Calibration Module**: Processes collected calibration data
- **Control Module**: Provides parameters for vibration suppression
- **Robot Interface**: Uses identified models for improved control

## Building

The C++ extensions are built automatically with the main project:

```bash
cmake -S . -B build
cmake --build build
```

## Dependencies

- **PyTorch**: Machine learning and tensor operations
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing
- **Eigen3**: C++ linear algebra
- **FFTW3**: Fast Fourier transforms
- **cnpy**: NumPy file I/O
- **pybind11**: Python bindings

## Performance

The C++ implementations provide significant performance improvements:
- **10-100x faster** than pure Python implementations
- **Memory efficient** processing of large datasets
- **Real-time capable** for online identification
- **Parallel processing** support for multi-core systems