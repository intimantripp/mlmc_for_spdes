#include <iostream>
#include "gbm_euler.hpp"
// #include "mlmc_core.hpp"
// #include "mlmc_test.hpp"

int main() {
    std::cout << "Starting MLMC simulation..." << std::endl;

    run_gbm_euler(10000);

    std::cout << "MLMC simulation completed." << std::endl;
    return 0;
}
