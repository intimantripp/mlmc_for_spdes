#include <iostream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "stoch_heat_eqn_energy_fe.hpp"
#include "stoch_heat_eqn_energy_cc.hpp"
#include "stoch_heat_eqn_energy_nn.hpp"


int main() {
    using namespace std::chrono;

    std::cout << "mlmc_cpp_omp running..." << std::endl;

    #ifdef _OPENMP
    std::cout << "OpenMP enabled. Max threads = " << omp_get_max_threads() << std::endl;
    #else
    std::cout << "OpenMP NOT enabled (this build will run serially)." << std::endl;
    #endif

    auto start = high_resolution_clock::now();

    // run_stoch_heat_eqn_energy_cc(5000);
    run_stoch_heat_eqn_energy_fe(5000);
    run_stoch_heat_eqn_energy_nn(5000);

    auto end = high_resolution_clock::now();
    std::cout << "Done in " << duration<double>(end - start).count() << " s\n";
    return 0;
}
