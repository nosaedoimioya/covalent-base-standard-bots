#ifndef IDENTIFICATION_MAPFITTER_HPP
#define IDENTIFICATION_MAPFITTER_HPP

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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
inline torch::Tensor masked_mse(const torch::Tensor &target,
                                const torch::Tensor &prediction) {
    auto mask = target.ne(0.0).to(prediction.dtype());
    auto diff = (target - prediction) * mask;
    auto mse = diff.pow(2).sum() / (mask.sum() + 1e-8);
    return mse;
}

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
    ShaperNetImpl(int in_features, std::vector<int> hidden) {
        body = register_module("body", torch::nn::Sequential());
        int last = in_features;
        for (int h : hidden) {
            body->push_back(torch::nn::Linear(last, h));
            body->push_back(torch::nn::ReLU());
            last = h;
        }
        order_head = register_module("order", torch::nn::Linear(last, 1));
        mode_head  = register_module("modes", torch::nn::Linear(last, 4));
    }

    /**
     * @brief Forward pass producing order and modes.
     * @param x Input feature tensor.
     * @return Pair of tensors containing classification order and mode outputs.
     */
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = body->forward(x);
        auto order = torch::sigmoid(order_head->forward(x));
        auto modes = mode_head->forward(x);
        return {order, modes};
    }
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
    MapFitter(int axes, int input_features, std::vector<int> hidden)
        : axes_(axes), in_features_(input_features), hidden_(std::move(hidden)) {
        for (int i = 0; i < axes_; ++i) {
            models_.push_back(std::make_shared<ShaperNet>(in_features_, hidden_));
        }
    }

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
               double lr = 1e-3) {
        for (int axis = 0; axis < axes_; ++axis) {
            auto model = models_[axis];
            model->train();
            torch::optim::Adam optim(model->parameters(), lr);
            auto bce = torch::nn::BCELoss();
            for (int e = 0; e < epochs; ++e) {
                auto out = model->forward(features[axis]);
                auto loss_cls = bce(out.first, orders[axis]);
                auto loss_reg = masked_mse(modes[axis], out.second * masks[axis]);
                auto loss = loss_cls + 10.0 * loss_reg;
                optim.zero_grad();
                loss.backward();
                optim.step();
            }
        }
    }

    /**
     * @brief Run inference for a given axis.
     * @param axis    Axis index.
     * @param feature Input feature tensor.
     * @return Pair of tensors containing predicted order and modes.
     */
    std::pair<torch::Tensor, torch::Tensor> infer(int axis,
                                                  const torch::Tensor &feature) {
        auto model = models_.at(axis);
        model->eval();
        return model->forward(feature);
    }

    /**
     * @brief Persist trained models to disk.
     * @param directory Destination directory.
     */
    void save_models(const std::string &directory) const {
        for (size_t i = 0; i < models_.size(); ++i) {
            std::ostringstream fn;
            fn << directory << "/axis_" << i << ".pt";
            torch::save(models_[i], fn.str());
        }
    }

    /**
     * @brief Load previously saved models from disk.
     * @param directory Source directory containing model files.
     */
    void load_models(const std::string &directory) {
        for (size_t i = 0; i < models_.size(); ++i) {
            std::ostringstream fn;
            fn << directory << "/axis_" << i << ".pt";
            torch::load(models_[i], fn.str());
        }
    }

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
                                           std::vector<int> hidden) {
        auto fitter = std::make_shared<MapFitter>(axes, input_features,
                                                  std::move(hidden));
        fitter->load_models(directory);
        return fitter;
    }
};

} // namespace identification

namespace py = pybind11;

// Pybind11 module exposing training and inference APIs
PYBIND11_MODULE(mapfitter, m) {
    py::class_<identification::MapFitter,
               std::shared_ptr<identification::MapFitter>>(m, "MapFitter")
        .def(py::init<int, int, std::vector<int>>(),
             py::arg("axes"), py::arg("input_features"), py::arg("hidden"))
        .def("train", &identification::MapFitter::train,
             py::arg("features"), py::arg("modes"), py::arg("orders"),
             py::arg("masks"), py::arg("epochs") = 200,
             py::arg("lr") = 1e-3)
        .def("infer", &identification::MapFitter::infer,
             py::arg("axis"), py::arg("feature"))
        .def("save_models", &identification::MapFitter::save_models)
        .def("load_models", &identification::MapFitter::load_models);

    py::class_<identification::ModelLoader>(m, "ModelLoader")
        .def_static("load", &identification::ModelLoader::load,
                    py::arg("directory"), py::arg("axes"),
                    py::arg("input_features"), py::arg("hidden"));
}

#endif // IDENTIFICATION_MAPFITTER_HPP