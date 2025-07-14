// filepath: /mlmc_cpp/mlmc_cpp/src/mlmc_core.c
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>   // for std::accumulate
#include <cassert>
#include "mlmc.hpp"
#include "gbm_euler.hpp"

void mlmc() {
    
    std::cout << "Running main MLMC method..." << std::endl;
    std::cout << "Call GBM Euler level function" << std::endl;
    gbm_l();

    std::cout << "Main MLMC method completed" << std::endl;

}


double regression(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = 0; i < n; ++i) {
        double log2y = std::log2(std::abs(y[i]));
        sum_x += x[i];
        sum_y += log2y;
        sum_xx += x[i] * x[i];
        sum_xy += x[i] * log2y;
    }
    double denom = n * sum_xx - sum_x * sum_x;
    double slope = (n * sum_xy - sum_x * sum_y) / denom;
    // Alpha is the negative slope
    return -slope;
}
