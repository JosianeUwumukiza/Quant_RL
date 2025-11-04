#include "AlphaEnv.hpp"
#include <algorithm>
#include <cmath>

AlphaEnv::AlphaEnv(int window_, double fee_bps_)
: window(window_), fee_bps(fee_bps_), t_idx(0), pos(0.0), eq(0.0), rng(123) {}

bool AlphaEnv::load_data(const std::string& csv_path) {
    return mr.load_csv(csv_path, true) && mr.size() > static_cast<size_t>(window+2);
}

void AlphaEnv::reset(unsigned int seed) {
    if (seed) rng.seed(seed);
    t_idx = window; // first obs index that has full window
    pos = 0.0;
    eq = 0.0;
}

double AlphaEnv::ret_at(size_t i) const {
    const auto& b0 = mr.at(i-1);
    const auto& b1 = mr.at(i);
    double r = (b1.close - b0.close) / std::max(1e-12, b0.close);
    return r;
}

double AlphaEnv::rolling_vol(size_t i, int w) const {
    if ((int)i < w+1) return 0.0;
    double mean=0.0; int n=0;
    for (int k=0;k<w;k++){ mean += ret_at(i-k); n++; }
    mean/=std::max(1,n);
    double var=0.0;
    for (int k=0;k<w;k++){ double d=ret_at(i-k)-mean; var+=d*d; }
    var/=std::max(1,n-1);
    return std::sqrt(std::max(1e-12,var));
}

double AlphaEnv::zscore_close(size_t i, int w) const {
    if ((int)i < w) return 0.0;
    double mean=0.0; for (int k=0;k<w;k++) mean += mr.at(i-k).close;
    mean /= w;
    double var=0.0; for (int k=0;k<w;k++){ double d = mr.at(i-k).close - mean; var += d*d; }
    var/=std::max(1,w-1);
    double sd = std::sqrt(std::max(1e-12,var));
    return sd>0 ? (mr.at(i).close - mean)/sd : 0.0;
}

std::vector<double> AlphaEnv::obs() const {
    std::vector<double> o; o.reserve(obs_dim());
    for (int k=window; k>=1; --k)
        o.push_back(ret_at(t_idx - k));
    o.push_back(rolling_vol(t_idx, std::min(window, 32)));
    o.push_back(zscore_close(t_idx, std::min(window, 32)));
    return o;
}

std::pair<double,bool> AlphaEnv::step(int action) {
    int a = std::clamp(action, -1, 1);
    // fee when changing position
    double fee = (std::abs(a - (int)pos)) * (fee_bps * 1e-4);
    // PnL from holding current pos over next bar
    double r = ret_at(t_idx+1);
    double pnl = pos * r - fee;
    eq += pnl;
    pos = (double)a;

    t_idx++;
    bool done = (t_idx+1 >= mr.size());
    return {pnl, done};
}
