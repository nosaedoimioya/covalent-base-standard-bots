#include "MapFitter.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace identification {

torch::Tensor masked_mse(const torch::Tensor &target,
                         const torch::Tensor &prediction) {
    auto mask = target.ne(0.0).to(prediction.dtype());
    auto diff = (target - prediction) * mask;
    auto mse = diff.pow(2).sum() / (mask.sum() + 1e-8);
    return mse;
}

ShaperNetImpl::ShaperNetImpl(int in_features, std::vector<int> hidden) {
    body = register_module("body", torch::nn::Sequential());
    int last = in_features;
    for (int h : hidden) {
        body->push_back(torch::nn::Linear(last, h));
        body->push_back(torch::nn::ReLU());
        last = h;
    }
    order_head = register_module("order", torch::nn::Linear(last, 1));
    mode_head = register_module("modes", torch::nn::Linear(last, 4));
}

std::pair<torch::Tensor, torch::Tensor> ShaperNetImpl::forward(torch::Tensor x) {
    x = body->forward(x);
    auto order = torch::sigmoid(order_head->forward(x));
    auto modes = mode_head->forward(x);
    return {order, modes};
}

MapFitter::MapFitter(int axes, int input_features, std::vector<int> hidden)
    : axes_(axes), in_features_(input_features), hidden_(std::move(hidden)) {
    for (int i = 0; i < axes_; ++i) {
        models_.emplace_back(in_features_, hidden_);
    }
}

void MapFitter::train(const std::vector<torch::Tensor> &features,
                      const std::vector<torch::Tensor> &modes,
                      const std::vector<torch::Tensor> &orders,
                      const std::vector<torch::Tensor> &masks,
                      int epochs,
                      double lr) {
    for (int axis = 0; axis < axes_; ++axis) {
        auto &model = models_[axis];
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

std::pair<torch::Tensor, torch::Tensor> MapFitter::infer(
    int axis, const torch::Tensor &feature) {
    auto &model = models_[axis];
    model->eval();
    return model->forward(feature);
}

void MapFitter::save_models(const std::string &directory) const {
    for (size_t i = 0; i < models_.size(); ++i) {
        std::ostringstream fn;
        fn << directory << "/axis_" << i << ".pt";
        torch::save(models_[i], fn.str());
    }
}

void MapFitter::load_models(const std::string &directory) {
    for (size_t i = 0; i < models_.size(); ++i) {
        std::ostringstream fn;
        fn << directory << "/axis_" << i << ".pt";
        torch::load(models_[i], fn.str());
    }
}

std::shared_ptr<MapFitter> ModelLoader::load(const std::string &directory,
                                             int axes,
                                             int input_features,
                                             std::vector<int> hidden) {
    auto fitter = std::make_shared<MapFitter>(axes, input_features,
                                              std::move(hidden));
    fitter->load_models(directory);
    return fitter;
}

} // namespace identification

namespace py = pybind11;

PYBIND11_MODULE(MapFitter, m) {
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