#include "mlcpp/data/csv.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mlcpp {

static std::vector<std::string> split(const std::string& line, char sep) {
    std::vector<std::string> out;
    std::string cell;
    std::istringstream ss(line);
    while (std::getline(ss, cell, sep)) out.push_back(cell);
    return out;
}

CsvData read_csv(const std::string& path, bool has_header) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("read_csv: không mở được file: " + path);

    CsvData out;
    std::string line;
    std::vector<std::vector<double>> rows;
    std::size_t ncols = 0;
    bool first = true;

    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF
        if (line.empty()) continue;

        if (first && has_header) {
            out.headers = split(line, ',');
            first = false;
            continue;
        }
        first = false;

        std::vector<std::string> cells = split(line, ',');
        std::vector<double> row;
        row.reserve(cells.size());
        for (const auto& c : cells) row.push_back(std::stod(c));

        if (ncols == 0)
            ncols = row.size();
        else if (row.size() != ncols)
            throw std::runtime_error("read_csv: số cột không đồng nhất");
        rows.push_back(std::move(row));
    }

    Matrix m(rows.size(), ncols, 0.0);
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < ncols; ++j) m(i, j) = rows[i][j];
    out.values = std::move(m);
    return out;
}

}  // namespace mlcpp
