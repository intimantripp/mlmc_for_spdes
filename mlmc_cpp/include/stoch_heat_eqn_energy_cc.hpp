#pragma once

#include <vector>

void run_stoch_heat_eqn_energy_cc(const int N = 10000);

std::pair<std::vector<double>, std::vector<double>> stoch_heat_eqn_energy_cc_l(int l, int N);
