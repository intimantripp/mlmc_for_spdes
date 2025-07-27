#pragma once

#include <vector>
#include <tuple>
#include <functional>

using LevelFunction = std::function<std::pair<std::vector<double>, std::vector<double>>(int, int)>;

std::tuple<double, std::vector<int>, std::vector<std::vector<double>>> mlmc(
    int N0,
    double eps,
    LevelFunction mlmc_l,
    double alpha_0,
    double beta_0,
    double gamma
);

double regression(const std::vector<int>& x, const std::vector<double>& y);
