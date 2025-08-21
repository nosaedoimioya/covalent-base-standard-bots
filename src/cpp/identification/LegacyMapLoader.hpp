#pragma once
#include <string>
#include <vector>
#include <torch/torch.h>

struct LegacyTensors {
    // Per-axis tensors
    std::vector<torch::Tensor> features; // [N, 3]  -> V_deg, R_mm, inertia
    std::vector<torch::Tensor> modes;    // [N, 4]  -> [wn1, log(z1+eps), wn2_or_0, log(z2+eps)_or_0]
    std::vector<torch::Tensor> orders;   // [N, 1]  -> has_second (0/1)
    std::vector<torch::Tensor> masks;    // [N, 4]  -> [1, 1, has2, has2]
};

/**
 * Load legacy Python .pkl CalibrationMap files and assemble per-axis training tensors.
 *
 * Each .pkl typically contains:
 *   - sineSweepDenominators: list(len=numPos) of list(len=axes) of denominator arrays
 *   - allV, allR, allInertia: [numPos, axes]
 *
 * The loader reproduces the old pipeline’s behavior:
 *   - Extract up to two complex-conjugate pole pairs from each denominator.
 *   - Compute (wn, zeta) per pair; keep primary always; second only if present.
 *   - Append features/targets per axis; skip samples with no complex pair at all.
 *   - If an axis ends up with 0 samples, insert a single masked dummy row to keep training code safe.
 */
LegacyTensors LoadLegacyTensorsFromPickle(const std::vector<std::string>& map_files,
                                          int axes);
