#include "LegacyMapLoader.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>

namespace py = pybind11;

static inline float to_f(double x) { return static_cast<float>(x); }

LegacyTensors LoadLegacyTensorsFromPickle(const std::vector<std::string>& map_files,
                                          int axes) {
    if (map_files.empty()) {
        throw std::runtime_error("No legacy .pkl maps provided.");
    }

    py::gil_scoped_acquire gil;
    py::module pickle = py::module::import("pickle");
    py::module np = py::module::import("numpy");

    // Python helpers: custom Unpickler that redirects legacy class paths,
    // and a function to extract (wn, zeta) modes from a denominator poly.
    const char* kHelper = R"PY(
import pickle, numpy as _np

class _RedirectingUnpickler(pickle.Unpickler):
    _targets = {
        ('src.identification.MapGenerationDelete', 'CalibrationMap',
         'build.src.cpp.identification.MapGeneration','CalibrationMap'),
    }
    def find_class(self, module, name):
        if (module, name) in self._targets:
            class CalibrationMap:  # empty shell; __dict__ restored by pickle
                pass
            return CalibrationMap
        return pickle.Unpickler.find_class(self, module, name)

def _load_legacy_pickle(path):
    with open(path, 'rb') as f:
        return _RedirectingUnpickler(f).load()

def _modes_from_den(a, real_tol=1e-4, zeta_min=0.01):
    # 'a' is a 1-D array-like of denominator coefficients (continuous-time)
    p = _np.roots(_np.asarray(a, dtype=float))
    modes = []
    used = _np.zeros(p.shape[0], dtype=bool)
    for i in range(p.shape[0]):
        if used[i]:
            continue
        pi = p[i]
        if abs(_np.imag(pi)) <= real_tol:
            continue
        # find its conjugate
        for j in range(i+1, p.shape[0]):
            if used[j]:
                continue
            if abs(p[j] - _np.conj(pi)) < real_tol:
                used[i] = True
                used[j] = True
                wn = abs(pi) / (2.0*_np.pi)          # Hz
                zeta = -_np.real(pi) / abs(pi)       # damping ratio
                if zeta >= zeta_min:
                    modes.append((float(wn), float(zeta)))
                break
    modes.sort(key=lambda m: m[0])
    return modes
)PY";

    py::dict env;
    env["__builtins__"] = py::module::import("builtins");  // optional but tidy
    py::exec(py::str(kHelper), env, env);

    py::object load_legacy_pickle = env["_load_legacy_pickle"];
    py::object modes_from_den    = env["_modes_from_den"];

    // Per-axis accumulators
    std::vector<std::vector<float>> X_by_axis(axes);     // flattened [V_deg, R_mm, inertia]
    std::vector<std::vector<float>> Y_by_axis(axes);     // flattened [wn1, log(z1+eps), wn2_or_0, log(z2+eps)_or_0]
    std::vector<std::vector<float>> ORD_by_axis(axes);   // flattened [has_second]
    std::vector<std::vector<float>> M_by_axis(axes);     // flattened [1,1,has2,has2]

    const double deg_per_rad = 180.0 / M_PI;
    const double mm_per_m = 1000.0;
    const double eps = 1e-6;

    for (const std::string& path : map_files) {
        py::object obj = load_legacy_pickle(py::str(path));

        // Arrays with shape [numPos, axes]
        py::array allV_arr       = obj.attr("allV").cast<py::array>();
        py::array allR_arr       = obj.attr("allR").cast<py::array>();
        py::array allInertia_arr = obj.attr("allInertia").cast<py::array>();

        if (allV_arr.ndim() != 2 || allR_arr.ndim() != 2 || allInertia_arr.ndim() != 2) {
            throw std::runtime_error("Legacy map arrays must be 2-D [numPos, axes]. File: " + path);
        }

        auto V = allV_arr.unchecked<double, 2>();        // read-only view
        auto R = allR_arr.unchecked<double, 2>();
        auto I = allInertia_arr.unchecked<double, 2>();

        const ssize_t numPos = V.shape(0);
        const ssize_t axN    = V.shape(1);
        if (axN != axes) {
            throw std::runtime_error("Axis count mismatch in legacy map: " + path);
        }

        // denominators: list(len=numPos) of list(len=axes) of arrays
        py::object denObj = obj.attr("sineSweepDenominators");
        if (!py::isinstance<py::list>(denObj) && !py::isinstance<py::tuple>(denObj)) {
            throw std::runtime_error("sineSweepDenominators must be a list. File: " + path);
        }
        py::sequence denSeq = denObj.cast<py::sequence>();
        if (static_cast<ssize_t>(denSeq.size()) != numPos) {
            throw std::runtime_error("sineSweepDenominators length != numPos. File: " + path);
        }

        // Iterate poses & axes
        for (ssize_t i = 0; i < numPos; ++i) {
            py::object perAxis = denSeq[i];
            py::sequence perAxisSeq = perAxis.cast<py::sequence>();
            if (static_cast<int>(perAxisSeq.size()) != axes) {
                throw std::runtime_error("sineSweepDenominators[pose] length != axes. File: " + path);
            }

            for (int a = 0; a < axes; ++a) {
                // Extract modes from denominator polynomial
                py::object den_a = perAxisSeq[a];
                py::list modes = modes_from_den(den_a).cast<py::list>();
                const ssize_t n_modes = py::len(modes);
                if (n_modes == 0) {
                    // Skip samples with no valid complex pair (parity with legacy)
                    continue;
                }

                // Primary mode
                py::tuple m1 = modes[0].cast<py::tuple>();
                double wn1  = m1[0].cast<double>();     // Hz
                double z1   = m1[1].cast<double>();

                // Optional second mode
                bool has_second = (n_modes > 1);
                double wn2 = 0.0, z2 = 0.0;
                if (has_second) {
                    py::tuple m2 = modes[1].cast<py::tuple>();
                    wn2  = m2[0].cast<double>();
                    z2   = m2[1].cast<double>();
                }

                // Features
                double v_deg  = V(i, a) * deg_per_rad;
                double r_mm   = R(i, a) * mm_per_m;
                double inertia= I(i, a);

                X_by_axis[a].push_back(to_f(v_deg));
                X_by_axis[a].push_back(to_f(r_mm));
                X_by_axis[a].push_back(to_f(inertia));

                // Targets (note: zeros for missing second mode to enable target-based masking)
                float z1_log = to_f(std::log(z1 + eps));
                float z2_log = has_second ? to_f(std::log(z2 + eps)) : 0.0f;

                Y_by_axis[a].push_back(to_f(wn1));
                Y_by_axis[a].push_back(z1_log);
                Y_by_axis[a].push_back(has_second ? to_f(wn2) : 0.0f);
                Y_by_axis[a].push_back(z2_log);

                // Order (classification)
                ORD_by_axis[a].push_back(has_second ? 1.0f : 0.0f);

                // Element-wise weights (used to zero predictions on missing 2nd-mode)
                M_by_axis[a].push_back(1.0f);
                M_by_axis[a].push_back(1.0f);
                M_by_axis[a].push_back(has_second ? 1.0f : 0.0f);
                M_by_axis[a].push_back(has_second ? 1.0f : 0.0f);
            }
        }
    }

    // Convert accumulators to per-axis tensors (ensure at least one row per axis)
    LegacyTensors out;
    out.features.reserve(axes);
    out.modes.reserve(axes);
    out.orders.reserve(axes);
    out.masks.reserve(axes);

    for (int a = 0; a < axes; ++a) {
        auto &X = X_by_axis[a];
        auto &Y = Y_by_axis[a];
        auto &O = ORD_by_axis[a];
        auto &M = M_by_axis[a];

        if (O.empty()) {
            // Create a single masked dummy row to keep training loop safe
            X.assign({0.f, 0.f, 0.f});
            Y.assign({0.f, 0.f, 0.f, 0.f});
            O.assign({0.f});
            M.assign({0.f, 0.f, 0.f, 0.f});
        }

        const int n = static_cast<int>(O.size());
        // X: [n,3]
        torch::Tensor tX = torch::from_blob(X.data(), {n, 3}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        // Y: [n,4]
        torch::Tensor tY = torch::from_blob(Y.data(), {n, 4}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        // O: [n,1]
        torch::Tensor tO = torch::from_blob(O.data(), {n, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        // M: [n,4]
        torch::Tensor tM = torch::from_blob(M.data(), {n, 4}, torch::TensorOptions().dtype(torch::kFloat32)).clone();

        out.features.push_back(std::move(tX));
        out.modes.push_back(std::move(tY));
        out.orders.push_back(std::move(tO));
        out.masks.push_back(std::move(tM));
    }

    return out;
}
