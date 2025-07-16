#include <iostream>
#include <chrono>
#include "gbm_euler.hpp"
// #include "mlmc_core.hpp"
// #include "mlmc_test.hpp"

int main() {
    using namespace std::chrono;

    std::cout << "Starting MLMC simulation..." << std::endl;
    auto start = high_resolution_clock::now();

    run_gbm_euler(150000);

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    std::cout << "MLMC simulation completed." << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;
    return 0;
}
