#pragma once
#include <utility>
#include <vector>


void run_dean_kawasaki_fe(const int N = 10000);

std::pair<std::vector<double>, std::vector<double>> dean_kawasaki_eqn_fe_l(int l, int N);
