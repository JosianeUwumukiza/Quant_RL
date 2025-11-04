#pragma once
#include <string>
#include <vector>

struct Bar {
    long ts;        // epoch seconds
    double open;
    double high;
    double low;
    double close;
    double volume;
};

class MarketReplay {
public:
    bool load_csv(const std::string& path, bool has_header=true);
    size_t size() const { return bars.size(); }
    const Bar& at(size_t i) const { return bars[i]; }
private:
    std::vector<Bar> bars;
};
