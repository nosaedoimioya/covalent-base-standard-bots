#pragma once

#include <string>
#include <cmath>

struct ProcessCalibrationOptions {
    std::string data_path;
    int poses;
    int axes;
    std::string robot_name = "test";
    std::string file_format = "csv";
    int num_joints = 6;
    double min_freq = 1.0;
    double max_freq = 60.0;
    double freq_space = 0.5;
    double max_disp = M_PI / 36.0;
    double dwell = 0.0;
    double Ts = 0.004;
    std::string sysid_type = "sine";
    std::string ctrl_config = "joint";
    double max_acc = 2.0;
    double max_vel = 18.0;
    int sine_cycles = 5;
    std::string sensor = "ToolAcc";
    int start_pose = 0;
    int max_map_size = 12;
    bool saved_maps = false;
};

int processCalibrationData(const ProcessCalibrationOptions &opts);
int processCalibrationDataCLI(int argc, const char **argv);