#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

// perf harness
#include "mlmc_test_performance.hpp"

// your MLMC level-callables
#include "stoch_heat_eqn_energy_nn.hpp"
#include "stoch_heat_eqn_energy_cc.hpp"
#include "stoch_heat_eqn_energy_fe.hpp"
#include "stoch_heat_eqn_fourier_modes_var.hpp"
#include "stoch_heat_eqn_fourier_modes_var_cc.hpp"
#include "stoch_heat_eqn_fourier_modes_var_fe.hpp"
#include "dean_kawasaki_nn.hpp"
#include "dean_kawasaki_cc.hpp"
#include "dean_kawasaki_fe.hpp"

struct PerfCase {
    std::string name; // used for folder/file names
    std::function<std::pair<std::vector<double>, std::vector<double>>(int,int)> mlmc_fn;
    double alpha, beta, gamma;
};

int main(int argc, char** argv) {
    using namespace std::chrono;

    // --- knobs you’ll tweak most often ---
    const int   M            = 8;        // refinement factor
    int         N0           = 100;     // base samples per level in mlmc()
    int         loops_per_eps= 5;        // repeats per epsilon
    std::vector<double> Eps  = {0.01, 0.001, 0.0001, 0.00001, 0.00001};

    // optional quick CLI: ./run_performance N0 loops
    if (argc >= 2) N0 = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) loops_per_eps = std::max(1, std::atoi(argv[2]));

    std::filesystem::create_directories("outputs_performance/timings");

    // ---- choose which cases to run by (un)commenting ----
    std::vector<PerfCase> cases = {
        // SHE Energy
        {"she_energy_nn", stoch_heat_eqn_energy_nn_l,  /*alpha*/2.0, /*beta*/2.0, /*gamma*/3.0},
        {"she_energy_cc", stoch_heat_eqn_energy_cc_l,  /*alpha*/2.0, /*beta*/2.0, /*gamma*/3.0},
        {"she_energy_fe", stoch_heat_eqn_energy_fe_l,  /*alpha*/2.0, /*beta*/4.0, /*gamma*/3.0}, // change to your analytics

        // SHE Squared amplitude (variance QoI)
        {"she_sqamp_nn",  stoch_heat_eqn_fourier_modes_var_l,    1.0, 2.0, 3.0},
        {"she_sqamp_cc",  stoch_heat_eqn_fourier_modes_var_cc_l, 1.0, 2.0, 3.0},
        {"she_sqamp_fe",  stoch_heat_eqn_fourier_modes_var_fe_l, 1.0, 3.0, 3.0},

        // Dean–Kawasaki
        {"dk_fe",         dean_kawasaki_eqn_fe_l,  2.0, 2.0, 3.0},
        {"dk_nn",         dean_kawasaki_eqn_nn_l,  2.0, 2.0, 3.0},
        {"dk_cc",         dean_kawasaki_eqn_cc_l,  2.0, 2.0, 3.0},
    };

    std::cout << "Starting performance runs...\n";
    auto T0 = high_resolution_clock::now();

    // run sequentially
    for (const auto& c : cases) {
        const std::string timings_dir   = "outputs_performance/timings/" + c.name;
        const std::string aggregate_csv = "outputs_performance/" + c.name + "_all_eps.csv";
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
    std::cout << "\nAll performance runs done in "
              << duration<double>(T1 - T0).count() << " s\n";
    return 0;
}
