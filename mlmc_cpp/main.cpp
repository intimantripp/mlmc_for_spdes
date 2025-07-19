#include <iostream>
#include <chrono>
#include "stoch_heat_eqn_energy.hpp"
#include "stoch_heat_eqn_fourier_modes.hpp"
#include "stoch_heat_eqn_fourier_modes_var.hpp"
#include "gbm_euler.hpp"

int main() {
    using namespace std::chrono;

    std::cout << "Starting MLMC simulation..." << std::endl;
    auto start = high_resolution_clock::now();

    // run_stoch_heat_eqn_fourier_modes(5000);
    run_stoch_heat_eqn_fourier_modes_var(1000);
    // run_stoch_heat_eqn_energy(5000);
    // run_gbm_euler();

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    std::cout << "MLMC simulation completed." << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;
    return 0;
}
