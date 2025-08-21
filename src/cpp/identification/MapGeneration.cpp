#include "MapGeneration.hpp"
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <complex>
#include <limits>
#include <iir/Butterworth.h>
#include <cnpy.h>
#include <Eigen/Eigen>

namespace {
    constexpr int    kButterOrder   = 4;
    constexpr double kButterCutoff  = 25.0;   // Hz, matches Python
}

// --- helper: second finite-difference derivative
static Eigen::VectorXd second_derivative(const Eigen::VectorXd& x, double Ts) {
    Eigen::VectorXd dd(x.size());
    dd.setZero();
    for (int i = 1; i < x.size()-1; ++i) {
        dd(i) = (x(i-1) - 2.0*x(i) + x(i+1)) / (Ts*Ts);
    }
    dd(0) = dd(1);
    dd(dd.size()-1) = dd(dd.size()-2);
    return dd;
}

// --- helper: least-squares analog invfreqs (matches util.Utility.invfreqs)
static void invfreqs_ls(const Eigen::VectorXcd& g, const Eigen::VectorXd& w,
                        int nB, int nA, const Eigen::VectorXd& wf,
                        Eigen::VectorXd& b, Eigen::VectorXd& a) {
    // Build linear system: [ H*D_a  -D_b ] [a_tail; b] = -H*d0
    // where D_a = [s^1, ..., s^nA], D_b = [1, s^1, ..., s^nB]
    const int N = g.size();
    const int n_unknown = nA + 1 + nB + 1 - 1; // a0 fixed=1 ⇒ unknowns: a1..a_nA, b0..b_nB
    Eigen::MatrixXcd M(N, n_unknown);
    Eigen::VectorXcd y(N);

    for (int i = 0; i < N; ++i) {
        std::complex<double> s(0.0, w(i));
        // left part: H * [s^1..s^nA]
        for (int j = 0; j < nA; ++j) M(i, j) = g(i) * std::pow(s, j+1);
        // right part: -[1, s^1..s^nB]
        for (int j = 0; j <= nB; ++j) M(i, nA + j) = -std::pow(s, j);
        y(i) = -g(i);
    }
    // weights
    Eigen::VectorXd W = (wf.size()==N ? wf : Eigen::VectorXd::Ones(N));
    Eigen::VectorXcd Wy = y;
    Eigen::MatrixXcd WM = M;
    for (int i=0;i<N;++i){ WM.row(i) *= W(i); Wy(i) *= W(i); }

    Eigen::VectorXcd x = WM.colPivHouseholderQr().solve(Wy);

    a = Eigen::VectorXd::Zero(nA+1); a(0) = 1.0;
    b = Eigen::VectorXd::Zero(nB+1);
    for (int j=0; j<nA; ++j) a(j+1) = x(j).real();
    for (int j=0; j<=nB; ++j) b(j) = x(nA + j).real();
}

// --- helper: poles from denominator (a[0..nA]), a0=1
static Eigen::VectorXcd poly_roots(const Eigen::VectorXd& a) {
    const int n = static_cast<int>(a.size()) - 1;
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n, n);
    C.block(1,0,n-1,n-1) = Eigen::MatrixXd::Identity(n-1, n-1);
    C.col(n-1) = -a.tail(n);
    Eigen::EigenSolver<Eigen::MatrixXd> es(C, /* computeEigenvectors = */ true);
    return es.eigenvalues();
}

// --- helper: project TCP accel → joint accel using Python Dynamics (mirrors Python)
static Eigen::VectorXd toolacc_to_jointacc(py::object robot_model,
                                           const Eigen::MatrixXd& tcp_acc, // [N x 3]
                                           const std::vector<double>& q0,
                                           int num_joints, int axis_commanded,
                                           bool gravity_comp) {
    const double g = 9.81;
    Eigen::Vector3d GRAV(0,0,g);

    // rotation matrix at q0
    py::object R_obj = robot_model.attr("get_rotation_matrix")(py::cast(q0));
    py::array_t<double, py::array::c_style | py::array::forcecast> R_py(R_obj);
    // Inspect buffer
    py::buffer_info rb = R_py.request();
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        R(static_cast<double*>(rb.ptr),
        static_cast<Eigen::Index>(rb.shape[0]),
        static_cast<Eigen::Index>(rb.shape[1]));

    // full Jacobian at q0 (6 x n); we use linear rows (0:3) and column axis_commanded-1
    py::object J_obj = robot_model.attr("get_jacobian_matrix")(py::cast(q0));
    py::array_t<double, py::array::c_style | py::array::forcecast> J_py(J_obj);
    py::buffer_info jb = J_py.request();
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        J(static_cast<double*>(jb.ptr),
        static_cast<Eigen::Index>(jb.shape[0]),
        static_cast<Eigen::Index>(jb.shape[1]));
    Eigen::Vector3d axis_col = J.block(0, axis_commanded-1, 3, 1); // linear part
    double axis_norm = axis_col.norm();
    if (axis_norm < 1e-12) axis_norm = 1.0;
    Eigen::Vector3d jac_tangent = axis_col / axis_norm;

    // frame selection: your pipeline uses world (we keep world; if 'tool', left-multiply by R^{-1})
    Eigen::VectorXd joint_acc(tcp_acc.rows());
    for (int k = 0; k < tcp_acc.rows(); ++k) {
        Eigen::Vector3d a_world = tcp_acc.row(k).transpose();
        if (gravity_comp) a_world -= GRAV;
        // project onto tangent / divide by |J_lin|
        double tang = a_world.dot(jac_tangent);
        joint_acc(k) = tang / axis_norm; // [rad/s^2]
    }
    return joint_acc;
}

static inline std::vector<double> to_row_major(const Eigen::MatrixXd& M) {
    std::vector<double> v;
    v.reserve(static_cast<size_t>(M.rows()) * static_cast<size_t>(M.cols()));
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        for (Eigen::Index j = 0; j < M.cols(); ++j) v.push_back(M(i, j));
    }
    return v;
}

static inline Eigen::MatrixXd from_row_major(const double* data,
                                             size_t rows, size_t cols) {
    Eigen::MatrixXd M(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                data[i * cols + j];
    return M;
}

CalibrationMap::CalibrationMap(int numPositions, int axesCommanded, int numJoints)
    : allWn_(Eigen::MatrixXd::Zero(numPositions, axesCommanded)),
      allZeta_(Eigen::MatrixXd::Zero(numPositions, axesCommanded)),
      allWn2_(Eigen::MatrixXd::Constant(numPositions, axesCommanded, std::nan("1"))),
      allZeta2_(Eigen::MatrixXd::Constant(numPositions, axesCommanded, std::nan("1"))),
      allInertia_(Eigen::MatrixXd::Zero(numPositions, axesCommanded)),
      allV_(Eigen::MatrixXd::Zero(numPositions, axesCommanded)),
      allR_(Eigen::MatrixXd::Zero(numPositions, axesCommanded)),
      num_positions_(numPositions),
      axes_commanded_(axesCommanded),
      num_joints_(numJoints) {}

void CalibrationMap::generate_from_pose(
    const Eigen::MatrixXd& q_cmd,              // [N x num_joints_]
    const Eigen::MatrixXd& tcp_acc,            // [N x 3]
    double input_Ts, double output_Ts,
    const std::vector<std::pair<int,int>>& segments,
    const std::vector<double>& freq_cmd,
    py::object robot_model,
    double V_angle, double R_radius,
    int axis_commanded,                         // 1-based
    const std::vector<double>& q0,              // size = num_joints_
    double max_freq_fit, bool gravity_comp,
    int store_row, int store_axis               // 0-based
){
    // Input joint acceleration (second derivative of commanded joint angle)
    Eigen::VectorXd q_axis = q_cmd.col(axis_commanded-1);
    Eigen::VectorXd qdd    = second_derivative(q_axis, input_Ts);

    // Optional low-pass (reuse your existing 4th-order Butterworth @ 25 Hz)
    Eigen::VectorXd qdd_filt = butterworthFilter(qdd, /*cutoff*/25.0, /*fs*/1.0/input_Ts);

    // Project TCP accel -> joint accel using Python Dynamics (matches your Python)
    Eigen::VectorXd joint_out = toolacc_to_jointacc(robot_model, tcp_acc, q0, num_joints_, axis_commanded, gravity_comp);

    // Build segment-wise FRFs
    std::vector<std::complex<double>> h_at_cmd;
    std::vector<double>               f_at_cmd;

    for (size_t k = 0; k < segments.size(); ++k) {
        auto [i0, i1] = segments[k];
        if (i0 < 0 || i1 <= i0 || i1 > qdd_filt.size() || i1 > joint_out.size()) continue;

        // FFT input
        int Ni = i1 - i0;
        Eigen::VectorXd in = qdd_filt.segment(i0, Ni);
        Eigen::VectorXcd IN = computeFFT(in);                       // size Ni/2
        Eigen::VectorXd  fin = Eigen::VectorXd::LinSpaced(IN.size(), 0, IN.size()-1)
                               * (1.0 / (Ni*input_Ts));

        // FFT output (same slice length)
        Eigen::VectorXd out = joint_out.segment(i0, Ni);
        Eigen::VectorXcd OUT = computeFFT(out);
        Eigen::VectorXd  fout = fin; // same length/sampling for the slice

        // FRF at commanded frequency -> pick nearest bin
        double fcmd = freq_cmd[k];
        int idx = (int)std::distance(fin.data(), std::lower_bound(fin.data(), fin.data()+fin.size(), fcmd));
        if (idx >= (int)fin.size()) idx = (int)fin.size()-1;
        if (idx < 0) idx = 0;
        std::complex<double> H = OUT(idx) / IN(idx);
        h_at_cmd.push_back(H);
        f_at_cmd.push_back(fcmd);
    }

    if (h_at_cmd.empty()) { return; }

    // Normalize & include "DC" sample as in Python
    Eigen::VectorXcd H_meas(h_at_cmd.size() + 1);
    Eigen::VectorXd  f_fit (h_at_cmd.size() + 1);
    H_meas(0) = std::complex<double>(1.0, 0.0);
    f_fit(0)  = 0.001; // tiny DC
    for (int i = 0; i < (int)h_at_cmd.size(); ++i) {
        H_meas(i+1) = h_at_cmd[i] / h_at_cmd[0];
        f_fit (i+1) = f_at_cmd[i];
    }
    // keep up to max_freq_fit
    std::vector<int> keep;
    for (int i=0;i<f_fit.size();++i) if (f_fit(i) <= max_freq_fit) keep.push_back(i);
    Eigen::VectorXcd Hk(keep.size());
    Eigen::VectorXd  fk(keep.size());
    for (int i=0;i<(int)keep.size();++i){ Hk(i)=H_meas(keep[i]); fk(i)=f_fit(keep[i]); }

    // Weights (DC heavy)
    Eigen::VectorXd W = Eigen::VectorXd::Ones(fk.size());
    if (W.size() > 0) W(0) = 1e6;

    // Grid search (nA=2..4, nB < nA)
    double best_err = std::numeric_limits<double>::infinity();
    Eigen::VectorXd best_a, best_b;
    for (int nA=2; nA<=4; ++nA) {
        for (int nB=0; nB<nA; ++nB) {
            Eigen::VectorXd a, b;
            invfreqs_ls(Hk, 2*M_PI*fk, nB, nA, W, b, a);       // <- order per Utility.invfreqs
            // quick error metric (no Control.matlab dependency)
            // Evaluate H_fit(jw) = B(jw)/A(jw)
            double err = 0.0;
            for (int i=0;i<fk.size();++i) {
                std::complex<double> s(0.0, 2*M_PI*fk(i));
                std::complex<double> num(0,0), den(1,0);
                for (int j=0;j<=nB;++j) num += b(j)*std::pow(s,j);
                for (int j=1;j<=nA;++j) den += a(j)*std::pow(s,j);
                std::complex<double> Hf = num/den;
                err += W(i) * std::norm(Hk(i) - Hf);
            }
            if (err < best_err) { best_err = err; best_a = a; best_b = b; }
        }
    }

    // Poles from best_a -> wn, zeta (dominant)
    Eigen::VectorXcd poles = poly_roots(best_a);
    // pick complex-conjugate with largest imag
    int idx_dom = 0;
    double best_im = 0.0;
    for (int i=0;i<poles.size();++i) {
        double im = std::abs(poles(i).imag());
        if (im > best_im) { best_im = im; idx_dom = i; }
    }
    std::complex<double> p = poles(idx_dom);
    double wn   = std::abs(p);
    double zeta = -p.real() / (wn + 1e-12);

    // Store
    allWn_     (store_row, store_axis) = wn;
    allZeta_   (store_row, store_axis) = zeta;

    // Call the Python method; pass q0 as a Python list
    py::object M_obj = robot_model.attr("get_mass_matrix")(py::cast(q0));

    // Force to a contiguous double array (will copy/cast if needed)
    py::array_t<double, py::array::c_style | py::array::forcecast> M_py(M_obj);

    // Inspect buffer
    py::buffer_info buf = M_py.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("get_mass_matrix() must return a 2D array");
    }
    const auto rows = static_cast<Eigen::Index>(buf.shape[0]);
    const auto cols = static_cast<Eigen::Index>(buf.shape[1]);

    // Map NumPy memory into an Eigen matrix (row-major matches NumPy)
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        M(static_cast<double*>(buf.ptr), rows, cols);

    // Optional safety check
    if (store_axis < 0 || store_axis >= cols) {
        throw std::out_of_range("store_axis out of range for mass matrix");
    }
    allInertia_(store_row, store_axis) = M(store_axis, store_axis);

    // V, R (caller provides these as part of your StructuredCalibrationData; set them there)
    allV_(store_row, store_axis) = V_angle; 
    allR_(store_row, store_axis) = R_radius;
}

void CalibrationMap::generateCalibrationMap(py::array_t<double> robot_input,
                                            py::array_t<double> robot_output,
                                            double input_Ts, double output_Ts,
                                            double max_freq_fit,
                                            bool gravity_comp,
                                            int shift_store_position) {
    // This is a simplified example algorithm.  Real implementation would
    // iterate over positions and axes similar to the Python version.
    auto in = robot_input.unchecked<1>();
    auto out = robot_output.unchecked<1>();

    Eigen::VectorXd in_vec(in.shape(0));
    Eigen::VectorXd out_vec(out.shape(0));
    for (ssize_t i = 0; i < in.shape(0); ++i) {
        in_vec(i) = in(i);
        out_vec(i) = out(i);
    }

    Eigen::VectorXd filt_in = butterworthFilter(in_vec, 25.0, 1.0/input_Ts);
    Eigen::VectorXcd in_fft = computeFFT(filt_in);
    Eigen::VectorXcd out_fft = computeFFT(out_vec);

    Eigen::VectorXd freq_in = Eigen::VectorXd::LinSpaced(in_fft.size(), 0, in_fft.size()-1) * (1.0/(in_fft.size()*input_Ts));
    Eigen::VectorXd freq_out = Eigen::VectorXd::LinSpaced(out_fft.size(), 0, out_fft.size()-1) * (1.0/(out_fft.size()*output_Ts));
    Eigen::VectorXcd out_interp = interpolateFFT(out_fft, freq_out, freq_in);
    Eigen::VectorXcd H = out_interp.cwiseQuotient(in_fft);

    Eigen::VectorXd b; Eigen::VectorXd a;
    fitTransferFunction(H, 2*M_PI*freq_in, 2, 2, b, a);
    if (b.size() > 0 && a.size() > 0) {
        allWn_(0,0) = std::abs(std::sqrt(a(1)));
        allZeta_(0,0) = -a(1) / (2*allWn_(0,0));
        // sineSweepNumerators_(0,0) = b(0);
        // sineSweepDenominators_(0,0) = a(0);
    }
}

Eigen::VectorXcd CalibrationMap::computeFFT(const Eigen::VectorXd &data) const {
    int N = static_cast<int>(data.size());
    fftw_complex *in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
    fftw_complex *out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
    for (int i=0;i<N;i++){in[i][0]=data(i); in[i][1]=0.0;}
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    Eigen::VectorXcd result(N/2);
    for (int i=0;i<N/2;i++){
        result(i) = std::complex<double>(out[i][0], out[i][1]);
    }
    fftw_destroy_plan(plan);
    fftw_free(in); fftw_free(out);
    return result;
}

Eigen::VectorXd CalibrationMap::butterworthFilter(const Eigen::VectorXd &data,
                                                  double cutoff, double fs) const {
    Iir::Butterworth::LowPass<4> filter;
    filter.setup(fs, cutoff);
    Eigen::VectorXd y(data.size());
    for (int i=0;i<data.size();++i) y(i) = filter.filter(data(i));
    return y;
}

Eigen::VectorXcd CalibrationMap::interpolateFFT(const Eigen::VectorXcd &data,
                                                const Eigen::VectorXd &freq_in,
                                                const Eigen::VectorXd &freq_out) const {
    Eigen::VectorXcd out(freq_out.size());
    for (int i=0;i<freq_out.size();++i){
        double f = freq_out(i);
        if (f <= freq_in(0)) { out(i) = data(0); continue; }
        if (f >= freq_in(freq_in.size()-1)) { out(i) = data(data.size()-1); continue; }
        auto it = std::lower_bound(freq_in.data(), freq_in.data()+freq_in.size(), f);
        int idx = int(it - freq_in.data()) - 1;
        double t = (f - freq_in(idx)) / (freq_in(idx+1) - freq_in(idx));
        out(i) = data(idx) * (1.0 - t) + data(idx+1) * t;
    }
    return out;
}

void CalibrationMap::fitTransferFunction(const Eigen::VectorXcd &H,
                                         const Eigen::VectorXd &w,
                                         int nB, int nA,
                                         Eigen::VectorXd &b,
                                         Eigen::VectorXd &a) const {
    int N = H.size();
    int M = nA + nB + 1;
    Eigen::MatrixXcd A(N, M);
    for (int i=0;i<N;i++) {
        std::complex<double> s = std::complex<double>(0, w(i));
        // denominator terms (skip a0=1)
        for (int j=0;j<nA;j++) {
            A(i,j) = -H(i) * std::pow(s, j+1);
        }
        // numerator terms
        for (int j=0;j<=nB;j++) {
            A(i,nA+j) = std::pow(s, j);
        }
    }
    Eigen::VectorXcd x = A.colPivHouseholderQr().solve(H);
    a = Eigen::VectorXd::Zero(nA+1);
    b = Eigen::VectorXd::Zero(nB+1);
    a(0) = 1.0;
    for (int j=0;j<nA;j++) a(j+1) = x(j).real();
    for (int j=0;j<=nB;j++) b(j) = x(nA+j).real();
}

void CalibrationMap::save_map(const std::string &filename) const {
    py::module pickle = py::module::import("pickle");
    py::object file = py::module::import("builtins").attr("open")(filename, "wb");
    pickle.attr("dump")(py::cast(*this), file);
}

CalibrationMap CalibrationMap::load_map(const std::string &filename) {
    py::module pickle = py::module::import("pickle");
    py::object file = py::module::import("builtins").attr("open")(filename, "rb");
    return pickle.attr("load")(file).cast<CalibrationMap>();
}

void CalibrationMap::save_npz(const std::string& filename) const {
    const size_t r = static_cast<size_t>(num_positions_);
    const size_t c = static_cast<size_t>(axes_commanded_);
    const std::vector<size_t> shape2d{r, c};

    // Flatten matrices row-major
    auto Wn    = to_row_major(allWn_);
    auto Zeta  = to_row_major(allZeta_);
    auto Wn2   = to_row_major(allWn2_);
    auto Zeta2 = to_row_major(allZeta2_);
    auto Iner  = to_row_major(allInertia_);
    auto V     = to_row_major(allV_);
    auto R     = to_row_major(allR_);

    // First array opens zip with mode "w", append the rest with "a"
    cnpy::npz_save(filename, "allWn",       Wn.data(),    shape2d, "w");
    cnpy::npz_save(filename, "allZeta",     Zeta.data(),  shape2d, "a");
    cnpy::npz_save(filename, "allWn2",      Wn2.data(),   shape2d, "a");
    cnpy::npz_save(filename, "allZeta2",    Zeta2.data(), shape2d, "a");
    cnpy::npz_save(filename, "allInertia",  Iner.data(),  shape2d, "a");
    cnpy::npz_save(filename, "allV",        V.data(),     shape2d, "a");
    cnpy::npz_save(filename, "allR",        R.data(),     shape2d, "a");

    // meta_sizes is a 1-D array with length 3
    int sizes[3] = { num_positions_, axes_commanded_, num_joints_ };
    const std::vector<size_t> shape1d{3};
    cnpy::npz_save(filename, "meta_sizes",  sizes,        shape1d, "a");
}

CalibrationMap CalibrationMap::load_npz(const std::string& filename) {
    cnpy::npz_t npz = cnpy::npz_load(filename);

    auto sizes_it = npz.find("meta_sizes");
    if (sizes_it == npz.end()) throw std::runtime_error("NPZ missing 'meta_sizes'");
    auto sizes = sizes_it->second;
    const int* sz = sizes.data<int>();
    int npos = sz[0], nax = sz[1], nj = sz[2];

    CalibrationMap m(npos, nax, nj);

    auto load2 = [&](const char* key, Eigen::MatrixXd& dst, bool required) {
        auto it = npz.find(key);
        if (it == npz.end()) {
            if (required) throw std::runtime_error(std::string("NPZ missing '") + key + "'");
            return;
        }
        auto arr = it->second;
        if (arr.shape.size() != 2 || arr.shape[0] != static_cast<size_t>(npos) ||
            arr.shape[1] != static_cast<size_t>(nax)) {
            throw std::runtime_error(std::string("NPZ array '") + key + "' has wrong shape");
        }
        dst = from_row_major(arr.data<double>(), arr.shape[0], arr.shape[1]);
    };

    load2("allWn", m.allWn_, true);
    load2("allZeta", m.allZeta_, true);
    load2("allWn2", m.allWn2_, false);
    load2("allZeta2", m.allZeta2_, false);
    load2("allInertia", m.allInertia_, true);
    load2("allV", m.allV_, true);
    load2("allR", m.allR_, true);

    return m;
}

PYBIND11_MODULE(MapGeneration, m) {
    py::class_<CalibrationMap>(m, "CalibrationMap")
        .def(py::init<int,int,int>())
        .def("num_positions", &CalibrationMap::num_positions)
        .def("axes_commanded", &CalibrationMap::axes_commanded)
        .def("num_joints", &CalibrationMap::num_joints)
        .def("generateCalibrationMap", &CalibrationMap::generateCalibrationMap,
            py::arg("robot_input"), py::arg("robot_output"),
            py::arg("input_Ts"), py::arg("output_Ts"),
            py::arg("max_freq_fit")=15.0,
            py::arg("gravity_comp")=false,
            py::arg("shift_store_position")=0)
        .def("generate_from_pose", &CalibrationMap::generate_from_pose,
            py::arg("q_cmd"),
            py::arg("tcp_acc"),
            py::arg("input_Ts"),
            py::arg("output_Ts"),
            py::arg("segments"),
            py::arg("freq_cmd"),
            py::arg("robot_model"),
            py::arg("V_angle")=0.0,
            py::arg("R_radius")=0.0,
            py::arg("axis_commanded")=1, // 1-based
            py::arg("q0")=std::vector<double>(),
            py::arg("max_freq_fit")=15.0,
            py::arg("gravity_comp")=false,
            py::arg("store_row")=0,
            py::arg("store_axis")=0) // 0-based
        // Expose matrices to Python
        .def_property_readonly("allWn",      [](const CalibrationMap& s){ return s.allWn(); })
        .def_property_readonly("allZeta",    [](const CalibrationMap& s){ return s.allZeta(); })
        .def_property_readonly("allWn2",     [](const CalibrationMap& s){ return s.allWn2(); })
        .def_property_readonly("allZeta2",   [](const CalibrationMap& s){ return s.allZeta2(); })
        .def_property_readonly("allInertia", [](const CalibrationMap& s){ return s.allInertia(); })
        .def_property_readonly("allV",       [](const CalibrationMap& s){ return s.allV(); })
        .def_property_readonly("allR",       [](const CalibrationMap& s){ return s.allR(); })
        .def("save_map", &CalibrationMap::save_map)
        .def_static("load_map", &CalibrationMap::load_map)
        .def("save_npz", &CalibrationMap::save_npz)
        .def_static("load_npz", &CalibrationMap::load_npz);
}