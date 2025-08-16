#ifndef IDENTIFICATION_MAPFITTER_HPP
#define IDENTIFICATION_MAPFITTER_HPP

#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

/**
 * @file MapFitter.hpp
 * @brief Neural network utilities for calibration map fitting.
 */

namespace identification {

/**
 * @brief Compute masked mean squared error, ignoring zero targets.
 *
 * @param target    Ground truth tensor where zeros are ignored.
 * @param prediction Model predictions.
 * @return Loss tensor representing the masked MSE.
 */
torch::Tensor masked_mse(const torch::Tensor &target,
                         const torch::Tensor &prediction);

/**
 * @brief Simple two head network used for fitting calibration maps.
 */
struct ShaperNetImpl : torch::nn::Module {
    torch::nn::Sequential body{nullptr};
    torch::nn::Linear order_head{nullptr};
    torch::nn::Linear mode_head{nullptr};
    
    /**
     * @brief Construct the neural network.
     * @param in_features Number of input features.
     * @param hidden      Sizes of hidden layers.
     */
    ShaperNetImpl(int in_features, std::vector<int> hidden);

    /**
     * @brief Forward pass producing order and modes.
     * @param x Input feature tensor.
     * @return Pair of tensors containing classification order and mode outputs.
     */
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(ShaperNet);

/**
 * @brief Driver for training and inference of per-axis models.
 */
class MapFitter {
public:
    /**
     * @brief Construct a MapFitter with the given model topology.
     * @param axes Number of joint axes.
     * @param input_features Number of features per sample.
     * @param hidden Hidden layer sizes for each model.
     */
    MapFitter(int axes, int input_features, std::vector<int> hidden);

    /**
     * @brief Train models for each axis.
     *
     * @param features Input feature tensors per axis.
     * @param modes    Target mode tensors per axis.
     * @param orders   Classification targets per axis.
     * @param masks    Masks indicating valid elements in mode tensors.
     * @param epochs   Training epochs.
     * @param lr       Learning rate.
     */
    void train(const std::vector<torch::Tensor> &features,
               const std::vector<torch::Tensor> &modes,
               const std::vector<torch::Tensor> &orders,
               const std::vector<torch::Tensor> &masks,
               int epochs = 200,
               double lr = 1e-3);

    /**
     * @brief Run inference for a given axis.
     * @param axis    Axis index.
     * @param feature Input feature tensor.
     * @return Pair of tensors containing predicted order and modes.
     */
    std::pair<torch::Tensor, torch::Tensor> infer(int axis,
                                                  const torch::Tensor &feature);

    /**
     * @brief Persist trained models to disk.
     * @param directory Destination directory.
     */
    void save_models(const std::string &directory) const;

    /**
     * @brief Load previously saved models from disk.
     * @param directory Source directory containing model files.
     */
    void load_models(const std::string &directory);

private:
    int axes_;
    int in_features_;
    std::vector<int> hidden_;
    std::vector<std::shared_ptr<ShaperNet>> models_;
};

/**
 * @brief Helper to load/save models.
 */
class ModelLoader {
public:
    /**
     * @brief Load models from disk and construct a MapFitter.
     * @param directory Directory containing saved models.
     * @param axes      Number of axes.
     * @param input_features Number of input features.
     * @param hidden    Hidden layer sizes.
     * @return Shared pointer to a MapFitter with loaded weights.
     */
    static std::shared_ptr<MapFitter> load(const std::string &directory,
                                           int axes,
                                           int input_features,
                                           std::vector<int> hidden);
};

} // namespace identification

#endif // IDENTIFICATION_MAPFITTER_HPP