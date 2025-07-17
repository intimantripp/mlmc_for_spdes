#pragma once

#include <vector>

void run_stoch_heat_eqn(const int N = 1000);

std::pair<std::vector<double>, std::vector<double>> stoch_heat_eqn_l(int l, int N);
