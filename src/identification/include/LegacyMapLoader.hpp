#pragma once
#include <string>
#include <vector>
#include <torch/torch.h>

/**
 * @file LegacyMapLoader.hpp
 * @brief Utilities for loading legacy Python calibration map files.
 * 
 * This module provides compatibility with legacy Python-generated calibration
 * maps stored in pickle format. It includes data structures and loading functions
 * that convert legacy map data into the format expected by the neural network
 * training pipeline.
 */

/**
 * @brief Container for per-axis training tensors extracted from calibration maps.
 * 
 * LegacyTensors organizes training data by robot axis, providing separate
 * tensors for features, target modes, classification orders, and validity masks.
 * This structure is used by the neural network training pipeline to organize
 * data for per-axis model training.
 */
struct LegacyTensors {
    // Per-axis tensors
    std::vector<torch::Tensor> features; ///< Input features [axes][N, 3] → [V_deg, R_mm, inertia]
    std::vector<torch::Tensor> modes;    ///< Target modes [axes][N, 4] → [wn1, log(z1+eps), wn2_or_0, log(z2+eps)_or_0]
    std::vector<torch::Tensor> orders;   ///< Classification targets [axes][N, 1] → has_second (0/1)
    std::vector<torch::Tensor> masks;    ///< Validity masks [axes][N, 4] → [1, 1, has2, has2]
};

/**
 * @brief Load legacy Python .pkl CalibrationMap files and assemble per-axis training tensors.
 *
 * This function provides backward compatibility with legacy Python-generated
 * calibration maps. It reads pickle files containing the old calibration map
 * format and converts them into the tensor format required by the neural network
 * training pipeline.
 *
 * Each .pkl typically contains:
 *   - sineSweepDenominators: list(len=numPos) of list(len=axes) of denominator arrays
 *   - allV, allR, allInertia: [numPos, axes] feature matrices
 *
 * The loader reproduces the old pipeline's behavior:
 *   - Extract up to two complex-conjugate pole pairs from each denominator.
 *   - Compute (wn, zeta) per pair; keep primary always; second only if present.
 *   - Append features/targets per axis; skip samples with no complex pair at all.
 *   - If an axis ends up with 0 samples, insert a single masked dummy row to keep training code safe.
 *
 * @param map_files Vector of paths to legacy .pkl calibration map files.
 * @param axes      Number of robot axes to process.
 * @return LegacyTensors structure containing organized training data for each axis.
 */
LegacyTensors LoadLegacyTensorsFromPickle(const std::vector<std::string>& map_files,
                                          int axes);
