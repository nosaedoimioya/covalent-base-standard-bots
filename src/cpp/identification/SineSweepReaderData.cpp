#include "SineSweepReader.hpp"

#include <cmath>

// Implements lightweight utilities that mirror missing pieces from the
// historical Python implementation.  These routines live in a separate source
// file so the header remains minimal and the logic can be unit tested via the
// pybind11 bindings.

std::size_t SineSweepReader::compute_num_maps() const {
    if (max_map_size_ == 0) {
        return 1;  // avoid division by zero; treat as single map
    }
    double runs = static_cast<double>(num_poses_) /
                  static_cast<double>(max_map_size_);
    std::size_t whole = static_cast<std::size_t>(runs);
    const double tol = 1e-6;
    if (runs - static_cast<double>(whole) > tol) {
        return whole + 1;
    }
    return whole;
}

std::vector<std::string> SineSweepReader::read_file(
    const std::string& filename) const {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() &&
            line.find_first_not_of(" \t\r\n") != std::string::npos) {
            lines.push_back(line);
        }
    }
    return lines;
}

std::vector<std::vector<double>> SineSweepReader::parse_data(
    const std::string& filename) const {
    std::vector<std::string> lines = read_file(filename);
    std::vector<std::vector<double>> parsed;
    for (const std::string& line : lines) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                row.push_back(std::stod(cell));
            }
        }
        if (!row.empty()) {
            parsed.push_back(std::move(row));
        }
    }
    return parsed;
}