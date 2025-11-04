#pragma once
#include <vector>

namespace Metrics {
    double sharpe(const std::vector<double>& pnl_series, double risk_free=0.0);
    double max_drawdown(const std::vector<double>& equity);
}
