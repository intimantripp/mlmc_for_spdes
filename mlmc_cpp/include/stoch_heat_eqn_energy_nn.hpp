#pragma once

#include <vector>

void run_stoch_heat_eqn_energy_nn(const int N = 10000);

std::pair<std::vector<double>, std::vector<double>> stoch_heat_eqn_energy_nn_l(int l, int N);
