#pragma once
#include <vector>
#include <functional>

void mlmc_test(
    std::function<std::pair<std::vector<double>, std::vector<double>>(int, int)> mlmc_fn,
    int M, int N, int L, int N0, const std::vector<double>& Eps,
    double validation_value = 0.0
);

double regression(const std::vector<int>& x, const std::vector<double>& y);

