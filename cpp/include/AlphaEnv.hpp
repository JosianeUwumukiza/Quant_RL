#pragma once
#include "MarketReplay.hpp"
#include <random>
#include <vector>

class AlphaEnv {
public:
    AlphaEnv(int window = 32, double fee_bps = 1.0);
    bool load_data(const std::string& csv_path);
    void reset(unsigned int seed=0);
    // action in {-1, 0, +1}
    std::pair<double,bool> step(int action);
    std::vector<double> obs() const;
    int obs_dim() const { return window + 2; }
    int action_dim() const { return 3; }
    size_t t() const { return t_idx; }
    double position() const { return pos; }
    double equity() const { return eq; }

private:
    MarketReplay mr;
    int window;
    double fee_bps;
    size_t t_idx;
    double pos;     // -1, 0, +1
    double eq;
    std::mt19937 rng;
    double ret_at(size_t i) const;
    double rolling_vol(size_t i, int w) const;
    double zscore_close(size_t i, int w) const;
};
