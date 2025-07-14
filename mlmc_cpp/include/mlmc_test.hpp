#pragma once
#include <vector>
#include <functional>
#include <string>

void mlmc_test(
    std::function<std::pair<std::vector<double>, std::vector<double>>(int, int)> mlmc_fn,
    int M, int N, int L, int N0, const std::vector<double>& Eps,
    const std::string& output_filename = "mlmc_convergence.csv"
);

double regression(const std::vector<int>& x, const std::vector<double>& y);

