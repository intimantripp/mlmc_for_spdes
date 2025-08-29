#include <iostream>
#include <chrono>
#include "stoch_heat_eqn_energy_nn.hpp"
#include "stoch_heat_eqn_energy_cc.hpp"
#include "stoch_heat_eqn_energy_fe.hpp"
#include "stoch_heat_eqn_fourier_modes.hpp"
#include "stoch_heat_eqn_fourier_modes_var.hpp"
#include "stoch_heat_eqn_fourier_modes_var_cc.hpp"
#include "stoch_heat_eqn_fourier_modes_var_fe.hpp"
#include "dean_kawasaki_nn.hpp"
#include "dean_kawasaki_cc.hpp"
#include "gbm_euler.hpp"

int main() {
    using namespace std::chrono;

    std::cout << "Starting MLMC simulation..." << std::endl;
    auto start = high_resolution_clock::now();

    // run_stoch_heat_eqn_fourier_modes(5000);
    // run_stoch_heat_eqn_fourier_modes_var(10000);
    // run_stoch_heat_eqn_fourier_modes_var_cc(10000);
    // run_stoch_heat_eqn_fourier_modes_var_fe(10000);
    // run_stoch_heat_eqn_energy_nn(20000);
    // run_stoch_heat_eqn_energy_cc(20000);
    run_stoch_heat_eqn_energy_fe(10000);
    // run_gbm_euler(1000000);
    // run_dean_kawasaki_nn(1000);
    // run_dean_kawasaki_cc(1000);

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    std::cout << "MLMC simulation completed." << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;
    return 0;
}
