#include <iomanip>
#include <string>
#include <sstream>

#include "wic/results.hpp"

namespace wic {

void write_csv_header(std::ostream& os) {
    os << "name,mean_us,stddev_us,throughput_gops,gpu,cuda_runtime\n";
}

void write_csv_row(std::ostream& os, const BenchResult& r, const std::string& gpu_name,
                   const std::string& cuda_version) {
    os << std::quoted(r.name) << ',' << std::setprecision(12) << r.mean_us << ',' << r.stddev_us << ','
       << r.throughput_gops << ',' << std::quoted(gpu_name) << ',' << std::quoted(cuda_version)
       << '\n';
}

}  // namespace wic
