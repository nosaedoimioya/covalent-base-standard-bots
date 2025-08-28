# Standard Bots Dynamic Calibration Guide

## Overview
Standard Bots robots can be dynamically calibrated to suppress vibration and improve positional accuracy. This repository provides all the components needed to collect calibration data, generate dynamic models, and integrate them into real‑time control.

The code base is organized into modules that mirror the calibration workflow:

| Module | Purpose |
| --- | --- |
| `src/calibration` | Run robot trajectories and record data |
| `src/identification` | Process recorded data to build dynamic models |
| `src/control` | Real‑time vibration suppression via input shaping |
| `src/robots` | Robot‑specific interfaces (Standard Bots implementation included) |
| `src/util` | Shared utilities for dynamics, trajectory generation, and data handling |

The guide below explains system requirements, repository layout, and step‑by‑step instructions for running a calibration with a Standard Bots robot.

## 1. System Setup
### Hardware
- Standard Bots robot with network connectivity
- Workstation PC running Ubuntu 20.04/22.04
- Wired LAN connection between PC and robot (≤1 ms round‑trip latency recommended)
- Optional tool‑center‑point (TCP) accelerometer for enhanced vibration measurements

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
- (Optional) Build tools for C++ modules if using the identification/control extensions

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
   python src/calibration/autoCalibration.py <robot_ip> <local_ip> \
       --name standard --type sine --freq 250 --ctrl joint
   ```
   Key options:
   - `--type` `bcb` or `sine`
   - `--freq` sampling frequency (Hz)
   - `--ctrl` `joint` or `task`
   - `--mdisp`, `--mvel`, `--macc` motion limits
   - `--nv`, `--nr` number of workspace angles and radii
3. Collected CSV files appear under `src/calibration/data/<date>/`.

### Processing Data
After a run:
```bash
python -m src.identification.python.process_calibration_data \
    src/calibration/data/<date> <poses> <axes> \
    --format csv --name standard --numjoints 6 \
    --minfreq 1 --maxfreq 20 --timestep 0.004
```
This generates calibration maps inside `src/calibration/models/`.

## 5. New Robot Integration
To support a different robot:
1. Create a subclass of `Robot` in `src/robots/` implementing the required interface methods (`move_home`, `command_robot`, etc.).
2. Add an entry in `RobotInterface._create_robot` so `--name` resolves to the new class.
3. Provide an accurate URDF under `src/robots/urdf/`.
4. Follow the skeleton in `src/robots/README.md` for detailed guidance.

## 6. Running & Collecting Data
- Keep the workspace clear and have an emergency stop available.
- Start with conservative motion limits and gradually increase.
- Monitor log output for dropped packets or limit violations.
- Each pose produces `robotData_motion_pose<i>_axis<j>.csv` and `robotData_static_pose<i>_axis<j>.csv` files.

## 7. Storing & Sending Data
1. After calibration, zip the relevant `data/<date>` directory:
   ```bash
   zip -r calibration_data.zip src/calibration/data/<date>
   ```
2. Email the archive to your Reforge contact for map generation or retain it for your own processing using the identification tools above.

## 8. Support
For questions about Standard Bots integration or calibration workflow, contact support@reforgerobotics.com.
