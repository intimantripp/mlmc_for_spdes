#pragma once
#include <functional>
#include <string>
#include <vector>

void mlmc_test_performance(
    std::function<std::pair<std::vector<double>, std::vector<double>>(int, int)> mlmc_fn,
    int M, 
    int N0,
    const std::vector<double>& Eps,
    double alpha, double beta, double gamma, 
    const std::string& timings_dir,          
    const std::string& aggregate_csv = "",
    int loops_per_eps = 10
);
