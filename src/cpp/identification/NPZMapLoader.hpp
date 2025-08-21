#pragma once
#include <string>
#include <vector>
#include <torch/torch.h>
#include "LegacyMapLoader.hpp"

/** Load .npz maps written by CalibrationMap::save_npz and produce per-axis tensors. */
LegacyTensors LoadNPZTensorsFromNPZ(const std::vector<std::string>& map_files,
                                    int axes);
