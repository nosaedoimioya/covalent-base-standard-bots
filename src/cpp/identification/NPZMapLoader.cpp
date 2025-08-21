#include "NPZMapLoader.hpp"
#include <cnpy.h>
#include <cmath>
#include <stdexcept>
#include <string>

static inline float f64(double x){ return static_cast<float>(x); }

LegacyTensors LoadNPZTensorsFromNPZ(const std::vector<std::string>& map_files,
                                    int axes) {
    if (map_files.empty()) throw std::runtime_error("No .npz maps provided.");

    std::vector<std::vector<float>> X_by_axis(axes);
    std::vector<std::vector<float>> Y_by_axis(axes);
    std::vector<std::vector<float>> O_by_axis(axes);
    std::vector<std::vector<float>> M_by_axis(axes);

    const double deg_per_rad = 180.0 / M_PI;
    const double mm_per_m    = 1000.0;
    const double eps         = 1e-6;

    for (const std::string& path : map_files) {
        cnpy::npz_t npz = cnpy::npz_load(path);

        auto get = [&](const char* key) -> cnpy::NpyArray& {
            auto it = npz.find(key);
            if (it == npz.end()) throw std::runtime_error(std::string("NPZ missing '")+key+"' in "+path);
            return it->second;
        };

        auto& meta = get("meta_sizes");
        const int* sz = meta.data<int>();
        const int rows = sz[0], cols = sz[1]; // poses, axes

        auto& Wn      = get("allWn");
        auto& Zeta    = get("allZeta");
        auto  Wn2_it  = npz.find("allWn2");
        auto  Z2_it   = npz.find("allZeta2");
        auto& Inertia = get("allInertia");
        auto& V       = get("allV");
        auto& R       = get("allR");

        const double* wn    = Wn.data<double>();
        const double* zeta  = Zeta.data<double>();
        const double* wn2   = (Wn2_it != npz.end()) ? Wn2_it->second.data<double>() : nullptr;
        const double* zeta2 = (Z2_it  != npz.end()) ? Z2_it->second.data<double>() : nullptr;
        const double* inertia = Inertia.data<double>();
        const double* v     = V.data<double>();
        const double* r     = R.data<double>();

        if (cols != axes) throw std::runtime_error("Axis count mismatch in "+path);

        for (int i = 0; i < rows; ++i) {
            for (int a = 0; a < axes; ++a) {
                const size_t idx = static_cast<size_t>(i)*axes + a;

                double wn1 = wn[idx];
                double z1  = zeta[idx];

                if (!(wn1 > 0.0) || !(z1 > 0.0)) {
                    // skip invalid primary mode
                    continue;
                }

                const double v_deg = v[idx] * deg_per_rad;
                const double r_mm  = r[idx] * mm_per_m;
                const double J     = inertia[idx];

                X_by_axis[a].push_back(f64(v_deg));
                X_by_axis[a].push_back(f64(r_mm));
                X_by_axis[a].push_back(f64(J));

                bool has2 = false;
                double wn22 = 0.0, z2 = 0.0;
                if (wn2 && zeta2) {
                    wn22 = wn2[idx];
                    z2   = zeta2[idx];
                    has2 = std::isfinite(wn22) && std::isfinite(z2) && wn22 > 0.0 && z2 > 0.0;
                }

                Y_by_axis[a].push_back(f64(wn1));
                Y_by_axis[a].push_back(f64(std::log(z1 + eps)));
                Y_by_axis[a].push_back(has2 ? f64(wn22) : 0.0f);
                Y_by_axis[a].push_back(has2 ? f64(std::log(z2 + eps)) : 0.0f);

                O_by_axis[a].push_back(has2 ? 1.0f : 0.0f);

                M_by_axis[a].push_back(1.0f);
                M_by_axis[a].push_back(1.0f);
                M_by_axis[a].push_back(has2 ? 1.0f : 0.0f);
                M_by_axis[a].push_back(has2 ? 1.0f : 0.0f);
            }
        }
    }

    LegacyTensors out;
    out.features.reserve(axes);
    out.modes.reserve(axes);
    out.orders.reserve(axes);
    out.masks.reserve(axes);

    for (int a = 0; a < axes; ++a) {
        auto &X = X_by_axis[a], &Y = Y_by_axis[a], &O = O_by_axis[a], &M = M_by_axis[a];
        if (O.empty()) {
            X.assign({0.f,0.f,0.f});
            Y.assign({0.f,0.f,0.f,0.f});
            O.assign({0.f});
            M.assign({0.f,0.f,0.f,0.f});
        }
        const int n = static_cast<int>(O.size());
        out.features.push_back(torch::from_blob(X.data(), {n,3},  torch::kFloat32).clone());
        out.modes   .push_back(torch::from_blob(Y.data(), {n,4},  torch::kFloat32).clone());
        out.orders  .push_back(torch::from_blob(O.data(), {n,1},  torch::kFloat32).clone());
        out.masks   .push_back(torch::from_blob(M.data(), {n,4},  torch::kFloat32).clone());
    }

    return out;
}
