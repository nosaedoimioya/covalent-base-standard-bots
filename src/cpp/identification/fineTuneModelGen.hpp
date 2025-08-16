#pragma once
#include <string>
#include <vector>

/**
 * @file fineTuneModelGen.hpp
 * @brief Interfaces for fine-tuning saved shaper neural network models
 */

/**
 * @brief Fine-tune previously saved shaper neural network models.
 *
 * This minimal implementation mirrors the behaviour of the earlier dummy
 * executable.  In a production system the models and map fitter would be
 * provided by the identification library.
 *
 * @param model_file Path to the directory containing existing models.
 * @param maps       Calibration map pickle files providing new data.
 * @param epochs     Number of fine-tuning epochs to run.
 * @param lr         Learning rate for optimisation.
 * @param save_file  Location to write the updated models.  If empty, a
 *                   default name is used.
 * @return Status code, 0 on success.
 */
int runFineTuneModelGen(const std::string &model_file,
                        const std::vector<std::string> &maps,
                        int epochs = 50,
                        double lr = 1e-4,
                        const std::string &save_file = "");