#pragma once

#include <chrono>
#include <cstdint>
#include <ostream>
#include <string>

namespace wic {

struct BenchResult {
    std::string name;
    double mean_us = 0.0;
    double stddev_us = 0.0;
    double throughput_gops = 0.0;  // illustrative; derived from workload def
};

void write_csv_header(std::ostream& os);
void write_csv_row(std::ostream& os, const BenchResult& r, const std::string& gpu_name,
                   const std::string& cuda_version, const std::string& git_sha);

inline double us_since(std::chrono::steady_clock::time_point t0) {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now() - t0).count();
}

}  // namespace wic
