#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <functional>

#include "mlmc_test_performance.hpp"
#include "stoch_heat_eqn_energy_nn.hpp"
#include "stoch_heat_eqn_energy_cc.hpp"
#include "stoch_heat_eqn_energy_fe.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif


struct PerfCase {
    std::string name;
    std::function<std::pair<std::vector<double>, std::vector<double>>(int,int)> mlmc_fn;
    double alpha, beta, gamma;
};

int main(int argc, char** argv) {
    using namespace std::chrono;

    #ifdef _OPENMP
    std::cout << "run_performance_omp: OpenMP enabled. Max threads = "
              << omp_get_max_threads() << "\n";
    #else
    std::cout << "run_performance_omp: OpenMP NOT enabled (will run serially).\n";
    #endif

    // Tunables
    const int   M  = 8;
    int         N0 = 100;
    int         loops_per_eps = 10;
    std::vector<double> Eps   = {0.0005, 0.001, 0.005, 0.01, 0.05};

    // Optional CLI: ./run_performance_omp N0 loops
    if (argc >= 2) N0 = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) loops_per_eps = std::max(1, std::atoi(argv[2]));

    // Separate outputs root so results don’t collide with serial runs
    const std::string root = "outputs_performance_omp";
    std::filesystem::create_directories(root + "/timings");

    // Choose OMP cases you actually have implemented
    std::vector<PerfCase> cases = {
        // SHE Energy (OMP versions)
        {"she_energy_nn_omp", stoch_heat_eqn_energy_nn_l,  2.0, 2.0, 3.0},
        {"she_energy_cc_omp", stoch_heat_eqn_energy_cc_l,  2.0, 2.0, 3.0},
        {"she_energy_fe_omp", stoch_heat_eqn_energy_fe_l,  2.0, 3.0, 3.0},
        // Add more once you have OMP variants:
        // {"she_sqamp_nn_omp",  stoch_heat_eqn_fourier_modes_var_l,    1.0, 2.0, 3.0},
        // {"she_sqamp_cc_omp",  stoch_heat_eqn_fourier_modes_var_cc_l, 1.0, 2.0, 3.0},
        // {"she_sqamp_fe_omp",  stoch_heat_eqn_fourier_modes_var_fe_l, 1.0, 3.0, 3.0},
        // {"dk_nn_omp", dean_kawasaki_eqn_nn_l, 2.0, 2.0, 3.0}, etc.
    };

    std::cout << "Starting OMP performance runs...\n";
    auto T0 = high_resolution_clock::now();

    for (const auto& c : cases) {
        const std::string timings_dir   = root + "/timings/" + c.name;
        const std::string aggregate_csv = root + "/" + c.name + "_all_eps.csv";
        std::filesystem::create_directories(timings_dir);

        std::cout << "\n=== Case: " << c.name
                  << " | N0=" << N0
                  << " | loops=" << loops_per_eps
                  << " | Eps size=" << Eps.size() << " ===\n";

        mlmc_test_performance(
            c.mlmc_fn,
            M, N0,
            Eps,
            c.alpha, c.beta, c.gamma,
            timings_dir,
            aggregate_csv,
            loops_per_eps
        );
    }

    auto T1 = high_resolution_clock::now();
    std::cout << "\nAll OMP performance runs done in "
              << duration<double>(T1 - T0).count() << " s\n";
    return 0;
}
