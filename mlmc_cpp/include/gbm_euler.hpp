#pragma once

#include <vector>

void run_gbm_euler(const int N = 10000);

// gbm_l function
std::pair<std::vector<double>, std::vector<double>> gbm_l(
    int l, int N, double S0, double K, double T, double r, double sig
);
