#pragma once

#include <vector>

void run_stoch_heat_eqn_energy_fe(const int N = 10000);

std::pair<std::vector<double>, std::vector<double>> stoch_heat_eqn_energy_fe_l(int l, int N);
