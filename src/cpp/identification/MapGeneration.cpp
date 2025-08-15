#include "MapGeneration.hpp"
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <fstream>
#include <iir/Butterworth.h>

CalibrationMap::CalibrationMap(int numPositions, int axesCommanded, int numJoints)
    : sineSweepNumerators_(Eigen::MatrixXd::Zero(numPositions, axesCommanded)),
      sineSweepDenominators_(Eigen::MatrixXd::Zero(numPositions, axesCommanded)),
      allWn_(Eigen::MatrixXd::Zero(numPositions, axesCommanded)),
      allZeta_(Eigen::MatrixXd::Zero(numPositions, axesCommanded)) {}

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
        sineSweepNumerators_(0,0) = b(0);
        sineSweepDenominators_(0,0) = a(0);
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

PYBIND11_MODULE(MapGeneration, m) {
    py::class_<CalibrationMap>(m, "CalibrationMap")
        .def(py::init<int,int,int>())
        .def("generateCalibrationMap", &CalibrationMap::generateCalibrationMap,
             py::arg("robot_input"), py::arg("robot_output"),
             py::arg("input_Ts"), py::arg("output_Ts"),
             py::arg("max_freq_fit")=15.0,
             py::arg("gravity_comp")=false,
             py::arg("shift_store_position")=0)
        .def("save_map", &CalibrationMap::save_map)
        .def_static("load_map", &CalibrationMap::load_map);
}