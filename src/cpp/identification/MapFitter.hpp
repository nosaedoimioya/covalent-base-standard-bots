#ifndef IDENTIFICATION_MAPFITTER_HPP
#define IDENTIFICATION_MAPFITTER_HPP

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <string>
#include <vector>
#include <memory>
#include <sstream>

namespace identification {

// Masked mean square error loss ignoring zero targets
inline torch::Tensor masked_mse(const torch::Tensor &target,
                                const torch::Tensor &prediction) {
    auto mask = target.ne(0.0).to(prediction.dtype());
    auto diff = (target - prediction) * mask;
    auto mse = diff.pow(2).sum() / (mask.sum() + 1e-8);
    return mse;
}

// Simple two head network used for fitting the calibration map
struct ShaperNetImpl : torch::nn::Module {
    torch::nn::Sequential body{nullptr};
    torch::nn::Linear order_head{nullptr};
    torch::nn::Linear mode_head{nullptr};

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

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = body->forward(x);
        auto order = torch::sigmoid(order_head->forward(x));
        auto modes = mode_head->forward(x);
        return {order, modes};
    }
};
TORCH_MODULE(ShaperNet);

// MapFitter drives training/inference for each joint axis
class MapFitter {
public:
    MapFitter(int axes, int input_features, std::vector<int> hidden)
        : axes_(axes), in_features_(input_features), hidden_(std::move(hidden)) {
        for (int i = 0; i < axes_; ++i) {
            models_.push_back(std::make_shared<ShaperNet>(in_features_, hidden_));
        }
    }

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

    std::pair<torch::Tensor, torch::Tensor> infer(int axis,
                                                  const torch::Tensor &feature) {
        auto model = models_.at(axis);
        model->eval();
        return model->forward(feature);
    }

    void save_models(const std::string &directory) const {
        for (size_t i = 0; i < models_.size(); ++i) {
            std::ostringstream fn;
            fn << directory << "/axis_" << i << ".pt";
            torch::save(models_[i], fn.str());
        }
    }

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

// Helper that mirrors the Python load_models/save_models workflow
class ModelLoader {
public:
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