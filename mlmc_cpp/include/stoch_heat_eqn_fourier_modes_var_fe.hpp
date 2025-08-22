#pragma once

#include <vector>

void run_stoch_heat_eqn_fourier_modes_var_fe(const int N = 20000);

std::pair<std::vector<double>, std::vector<double>> stoch_heat_eqn_fourier_modes_var_fe_l(int l, int N);
