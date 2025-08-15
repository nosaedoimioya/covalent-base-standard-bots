#pragma once
#include <vector>
#include <string>

/**
 * @brief Entry point for fine-tuning saved shaper neural network models.
 *
 * Command line options:
 *  --model <file>   Location of existing models to load.
 *  --maps <files>   Calibration map files providing new training data.
 *  --epochs <int>   Number of training epochs (default: 50).
 *  --lr <double>    Learning rate for fine tuning (default: 1e-4).
 *  --save <file>    Output location for updated models (default: fine_tuned_map).
 */
int runFineTuneModelGen(int argc, char** argv);