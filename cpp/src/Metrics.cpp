#include "Metrics.hpp"
#include <cmath>
#include <algorithm>

double Metrics::sharpe(const std::vector<double>& pnl, double rf) {
    if (pnl.size() < 2) return 0.0;
    double mean = 0.0;
    for (auto x : pnl) mean += x;
    mean /= pnl.size();
    double var = 0.0;
    for (auto x : pnl) var += (x - mean)*(x - mean);
    var = (pnl.size() > 1) ? var / (pnl.size()-1) : 0.0;
    double sd = std::sqrt(std::max(1e-12, var));
    double excess = mean - rf;
    return sd > 0 ? excess / sd : 0.0;
}

double Metrics::max_drawdown(const std::vector<double>& eq) {
    double peak = -1e18, mdd = 0.0;
    for (auto v : eq) {
        peak = std::max(peak, v);
        mdd = std::max(mdd, (peak - v));
    }
    return mdd;
}
