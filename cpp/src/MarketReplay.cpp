#include "MarketReplay.hpp"
#include <fstream>
#include <sstream>

bool MarketReplay::load_csv(const std::string& path, bool has_header) {
    bars.clear();
    std::ifstream f(path);
    if (!f.good()) return false;
    std::string line;
    if (has_header && std::getline(f, line)) { /* skip header */ }
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        Bar b{};
        // Expected columns: ts,open,high,low,close,volume
        std::getline(ss, cell, ','); if(cell.empty()) continue; b.ts = std::stoll(cell);
        std::getline(ss, cell, ','); b.open = std::stod(cell);
        std::getline(ss, cell, ','); b.high = std::stod(cell);
        std::getline(ss, cell, ','); b.low  = std::stod(cell);
        std::getline(ss, cell, ','); b.close= std::stod(cell);
        std::getline(ss, cell, ','); b.volume= std::stod(cell);
        bars.push_back(b);
    }
    return !bars.empty();
}
