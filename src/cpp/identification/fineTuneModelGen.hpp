#pragma once
#include <string>
#include <vector>

/**
 * @file FineTuneModelGen.hpp
 * @brief Interfaces for fine-tuning saved shaper neural network models.
 * 
 * This module provides functionality for fine-tuning pre-trained neural network
 * models used in robot dynamics identification. It allows existing models to be
 * updated with new calibration data without requiring complete retraining from
 * scratch, enabling incremental model improvement and adaptation to new robot
 * configurations or operating conditions.
 */

/**
 * @brief Fine-tune previously saved shaper neural network models.
 *
 * This function provides the capability to fine-tune existing neural network
 * models with new calibration data. Fine-tuning allows models to adapt to new
 * robot configurations, operating conditions, or additional training data while
 * preserving the knowledge learned from previous training sessions.
 *
 * This minimal implementation mirrors the behaviour of the earlier dummy
 * executable. In a production system, the models and map fitter would be
 * provided by the identification library. The function loads existing models,
 * applies additional training epochs with new data, and saves the updated models.
 *
 * @param model_file Path to the directory containing existing trained models.
 * @param maps       Vector of calibration map file paths providing new training data.
 * @param epochs     Number of fine-tuning epochs to run (default: 50).
 * @param lr         Learning rate for optimization during fine-tuning (default: 1e-4).
 * @param save_file  Location to write the updated models. If empty, a default
 *                   name is used based on the input model file.
 * @return Status code, 0 on success, non-zero on error.
 */
int runFineTuneModelGen(const std::string &model_file,
                        const std::vector<std::string> &maps,
                        int epochs = 50,
                        double lr = 1e-4,
                        const std::string &save_file = "");