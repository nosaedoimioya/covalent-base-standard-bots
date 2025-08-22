#include "MapFitter.hpp"
#include <utility>
#include <sstream>

namespace identification {

// --- losses ---------------------------------------------------------------

torch::Tensor masked_mse(const torch::Tensor &target,
                         const torch::Tensor &prediction) {
    // Ignore elements where target == 0 (how we mark “missing second mode”).
    auto mask = target.ne(0.0).to(prediction.dtype());
    auto diff = (target - prediction) * mask;
    // avoid div by zero (all targets can be zeros in rare batches)
    auto denom = mask.sum().clamp_min(1e-8);
    return diff.pow(2).sum() / denom;
}

// --- model ----------------------------------------------------------------

ShaperNetImpl::ShaperNetImpl(int in_features, std::vector<int> hidden) {
    body = register_module("body", torch::nn::Sequential());
    int last = in_features;
    for (int h : hidden) {
        body->push_back(torch::nn::Linear(last, h));
        body->push_back(torch::nn::ReLU());
        last = h;
    }
    order_head = register_module("order", torch::nn::Linear(last, 1));  // binary
    mode_head  = register_module("modes", torch::nn::Linear(last, 4));  // [wn1, lnζ1, wn2, lnζ2]
}

std::pair<torch::Tensor, torch::Tensor>
ShaperNetImpl::forward(torch::Tensor x) {
    x = body->forward(x);
    auto order = torch::sigmoid(order_head->forward(x));
    auto modes = mode_head->forward(x);
    return {order, modes};
}

// --- driver ---------------------------------------------------------------

MapFitter::MapFitter(int axes, int input_features, std::vector<int> hidden)
    : axes_(axes), in_features_(input_features), hidden_(std::move(hidden)) {
    models_.reserve(axes_);
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
    // All tensors are expected per-axis:
    //   features[a] : [N, 3]          → [V_deg, R_mm, inertia]
    //   modes[a]    : [N, 4]          → [wn1, lnζ1, wn2, lnζ2], zeros for missing
    //   orders[a]   : [N] or [N,1]    → 0 or 1
    //   masks[a]    : [N, 4]          → 1 for valid element, else 0 (redundant with zero-target)
    for (int axis = 0; axis < axes_; ++axis) {
        auto &model = models_[axis];
        model->train();

        // Optimizer
        torch::optim::Adam optim(model->parameters(), torch::optim::AdamOptions(lr));

        // Targets / inputs
        auto X = features[axis].contiguous().to(torch::kFloat32);
        auto Y_modes = modes[axis].contiguous().to(torch::kFloat32);
        auto Y_order = orders[axis].contiguous().to(torch::kFloat32);
        auto W_mask  = masks[axis].contiguous().to(torch::kFloat32);

        if (Y_order.dim() == 1) {
            Y_order = Y_order.unsqueeze(1); // [N] → [N,1]
        }

        auto bce = torch::nn::BCELoss();

        for (int e = 0; e < epochs; ++e) {
            auto out = model->forward(X);
            auto pred_order = out.first;                  // [N,1]
            auto pred_modes = out.second;                 // [N,4]

            auto loss_cls = bce(pred_order, Y_order);
            // Either of these two lines alone is fine. Using both is extra-safe:
            //   – masked_mse ignores zero targets
            //   – multiplying prediction by mask zeroes gradients where target is missing
            auto loss_reg = masked_mse(Y_modes, pred_modes * W_mask);

            auto loss = loss_cls + 10.0 * loss_reg;

            optim.zero_grad();
            loss.backward();
            optim.step();
        }
    }
}

std::pair<torch::Tensor, torch::Tensor>
MapFitter::infer(int axis, const torch::Tensor &feature) {
    auto &model = models_.at(axis);
    model->eval();
    auto X = feature.contiguous().to(torch::kFloat32);
    return model->forward(X);
}

void MapFitter::save_models(const std::string &directory) const {
    for (size_t i = 0; i < models_.size(); ++i) {
        std::cout << "Saving model for axis " << i << " to " << directory<< "/axis_" << i << ".pt" << "\n";
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

// --- loader ---------------------------------------------------------------

std::shared_ptr<MapFitter> ModelLoader::load(const std::string &directory,
                                             int axes,
                                             int input_features,
                                             std::vector<int> hidden) {
    auto fitter = std::make_shared<MapFitter>(axes, input_features, std::move(hidden));
    fitter->load_models(directory);
    return fitter;
}

} // namespace identification
