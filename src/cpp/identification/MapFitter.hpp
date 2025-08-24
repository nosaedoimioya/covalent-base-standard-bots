#ifndef IDENTIFICATION_MAPFITTER_HPP
#define IDENTIFICATION_MAPFITTER_HPP

#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

/**
 * @file MapFitter.hpp
 * @brief Neural network utilities for calibration map fitting and robot dynamics identification.
 * 
 * This module provides a complete framework for training neural networks to predict
 * robot dynamics parameters from calibration data. It includes a custom neural network
 * architecture (ShaperNet) designed specifically for fitting calibration maps, along
 * with utilities for training, inference, and model persistence.
 */

namespace identification {

/**
 * @brief Compute masked mean squared error, ignoring zero targets.
 *
 * This loss function is specifically designed for handling missing data in calibration
 * maps where some samples may not have second-order dynamics (represented as zeros).
 * The function computes MSE only on non-zero target values and normalizes by the
 * number of valid elements to prevent bias from missing data.
 *
 * @param target    Ground truth tensor where zeros are ignored (missing second mode).
 * @param prediction Model predictions tensor.
 * @return Loss tensor representing the masked MSE.
 */
torch::Tensor masked_mse(const torch::Tensor &target,
                         const torch::Tensor &prediction);

/**
 * @brief Neural network architecture for fitting calibration maps.
 * 
 * ShaperNet is a custom neural network designed specifically for robot dynamics
 * identification. It features a shared body network followed by two specialized heads:
 * - Order head: Binary classification to predict if second-order dynamics are present
 * - Mode head: Regression to predict dynamics parameters (wn1, lnζ1, wn2, lnζ2)
 * 
 * The network takes robot configuration features (joint angle, radius, inertia) as
 * input and outputs both classification and regression predictions simultaneously.
 */
struct ShaperNetImpl : torch::nn::Module {
    torch::nn::Sequential body{nullptr};      ///< Shared feature extraction layers
    torch::nn::Linear order_head{nullptr};    ///< Binary classification head
    torch::nn::Linear mode_head{nullptr};     ///< Dynamics parameter regression head
    
    /**
     * @brief Construct the neural network with specified architecture.
     * 
     * Creates a feedforward network with ReLU activations and two output heads.
     * The body consists of configurable hidden layers, followed by specialized
     * heads for order classification and mode parameter regression.
     * 
     * @param in_features Number of input features (typically 3: V_deg, R_mm, inertia).
     * @param hidden      Sizes of hidden layers in the body network.
     */
    ShaperNetImpl(int in_features, std::vector<int> hidden);

    /**
     * @brief Forward pass producing order classification and mode predictions.
     * 
     * Processes input features through the shared body network, then applies
     * specialized heads to produce both classification and regression outputs.
     * The order output is sigmoid-activated for binary classification, while
     * the modes output is linear for regression.
     * 
     * @param x Input feature tensor of shape [batch_size, in_features].
     * @return Pair of tensors: (order_prediction, mode_predictions) where
     *         order_prediction has shape [batch_size, 1] and mode_predictions
     *         has shape [batch_size, 4] for [wn1, lnζ1, wn2, lnζ2].
     */
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(ShaperNet);

/**
 * @brief Main driver class for training and inference of per-axis dynamics models.
 * 
 * MapFitter manages a collection of ShaperNet models, one for each robot axis.
 * It provides a unified interface for training all models simultaneously using
 * calibration data, running inference on new configurations, and persisting
 * trained models to disk. The class handles the complete machine learning
 * pipeline from data preparation to model deployment.
 */
class MapFitter {
public:
    /**
     * @brief Construct a MapFitter with the specified model topology.
     * 
     * Initializes separate ShaperNet models for each robot axis with identical
     * architecture. All models share the same input features and hidden layer
     * configuration but are trained independently for each axis.
     * 
     * @param axes Number of joint axes in the robot.
     * @param input_features Number of input features per sample (typically 3).
     * @param hidden Hidden layer sizes for each model's body network.
     */
    MapFitter(int axes, int input_features, std::vector<int> hidden);

    /**
     * @brief Train models for each axis using provided calibration data.
     *
     * Performs supervised learning on all axes simultaneously. Each axis model
     * learns to predict dynamics parameters from robot configuration features.
     * The training uses a combined loss function with binary cross-entropy for
     * order classification and masked MSE for mode parameter regression.
     *
     * @param features Input feature tensors per axis [axes][N, 3] → [V_deg, R_mm, inertia].
     * @param modes    Target mode tensors per axis [axes][N, 4] → [wn1, lnζ1, wn2, lnζ2].
     * @param orders   Classification targets per axis [axes][N] → 0 or 1 (has second mode).
     * @param masks    Masks indicating valid elements in mode tensors [axes][N, 4].
     * @param epochs   Number of training epochs for all models.
     * @param lr       Learning rate for Adam optimizer.
     */
    void train(const std::vector<torch::Tensor> &features,
               const std::vector<torch::Tensor> &modes,
               const std::vector<torch::Tensor> &orders,
               const std::vector<torch::Tensor> &masks,
               int epochs = 200,
               double lr = 1e-3);

    /**
     * @brief Run inference for a specific axis using trained model.
     * 
     * Performs forward pass through the trained model for the specified axis
     * to predict dynamics parameters for new robot configurations.
     * 
     * @param axis    Axis index (0-based).
     * @param feature Input feature tensor of shape [batch_size, in_features].
     * @return Pair of tensors containing predicted order and modes for the axis.
     */
    std::pair<torch::Tensor, torch::Tensor> infer(int axis,
                                                  const torch::Tensor &feature);

    /**
     * @brief Persist trained models to disk for later use.
     * 
     * Saves each axis model to a separate file in the specified directory.
     * Models are saved in PyTorch's standard format and can be loaded later
     * for inference or continued training.
     * 
     * @param directory Destination directory for model files.
     */
    void save_models(const std::string &directory) const;

    /**
     * @brief Load previously saved models from disk.
     * 
     * Restores model weights from saved files. The directory should contain
     * model files for all axes in the expected naming format.
     * 
     * @param directory Source directory containing saved model files.
     */
    void load_models(const std::string &directory);

private:
    int axes_;                              ///< Number of robot axes
    int in_features_;                       ///< Number of input features per sample
    std::vector<int> hidden_;               ///< Hidden layer sizes for each model
    std::vector<ShaperNet> models_;         ///< Trained models, one per axis
};

/**
 * @brief Utility class for loading and managing trained MapFitter models.
 * 
 * Provides static methods to create MapFitter instances from saved model files.
 * This class simplifies the process of deploying trained models in production
 * environments by handling the model loading and initialization.
 */
class ModelLoader {
public:
    /**
     * @brief Load models from disk and construct a ready-to-use MapFitter.
     * 
     * Creates a new MapFitter instance and loads trained weights from the
     * specified directory. The returned instance is ready for inference
     * without requiring additional training.
     * 
     * @param directory Directory containing saved model files.
     * @param axes      Number of axes (must match saved models).
     * @param input_features Number of input features (must match saved models).
     * @param hidden    Hidden layer sizes (must match saved models).
     * @return Shared pointer to a MapFitter with loaded weights ready for inference.
     */
    static std::shared_ptr<MapFitter> load(const std::string &directory,
                                           int axes,
                                           int input_features,
                                           std::vector<int> hidden);
};

} // namespace identification

#endif // IDENTIFICATION_MAPFITTER_HPP