#pragma once
#include <string>
#include <vector>
#include <torch/torch.h>
#include "LegacyMapLoader.hpp"

/**
 * @file NPZMapLoader.hpp
 * @brief Utilities for loading NumPy compressed archive (.npz) calibration maps.
 * 
 * This module provides functionality for loading calibration maps stored in
 * NumPy compressed archive format (.npz files). It converts the loaded data
 * into the tensor format required by the neural network training pipeline,
 * providing compatibility with modern Python-generated calibration maps.
 */

/**
 * @brief Load .npz maps written by CalibrationMap::save_npz and produce per-axis tensors.
 * 
 * This function loads calibration maps stored in NumPy compressed archive format
 * (.npz files) and converts them into the LegacyTensors format expected by the
 * neural network training pipeline. It provides compatibility with modern Python
 * tools that generate calibration maps using the CalibrationMap::save_npz method.
 * 
 * The function processes the same data structures as the legacy pickle loader
 * but works with the more portable and efficient .npz format. It extracts
 * dynamics parameters and features from the compressed archive and organizes
 * them by robot axis for training.
 * 
 * @param map_files Vector of paths to .npz calibration map files.
 * @param axes      Number of robot axes to process.
 * @return LegacyTensors structure containing organized training data for each axis.
 */
LegacyTensors LoadNPZTensorsFromNPZ(const std::vector<std::string>& map_files,
                                    int axes);
