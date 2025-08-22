#include "stoch_heat_eqn_energy_cc.hpp"
#include "mlmc_test.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

// Define pi if it isn't defined
#ifndef M_PI
constexpr double M_PI = std::acos(-1.0);
#endif

// Define Fourier Mode
static const int mode = 1;

// Global RNG
static std::mt19937 RNG{ std::random_device{}() };
static std::normal_distribution<> Z(0.0, 1.0);

// Utility to index a flattened 2D array: xi(i, n) = i * N2 + n
inline int idx(int i, int n, int N2) {return i * N2 + n; }

void run_stoch_heat_eqn_energy_cc(const int N) {
    std::cout << "Running MLMC Stochastic Heat Equation - Energy CC\n" << std::endl;
    int M = 8;
    int L = 6;
    int N0 = 100;
    std::vector<double> Eps = {0.00025, 0.0005, 0.001, 0.005, 0.01};

    std::string output_complexity_filename = "../outputs/mlmc_complexity_stoch_heat_eqn_energy_cc.csv";
    std::string output_convergence_filename = "../outputs/mlmc_convergence_stoch_heat_eqn_energy_cc.csv";
    std::string output_regression_filename = "../outputs/mlmc_regression_stoch_heat_eqn_energy_cc.csv";
    

    mlmc_test(
        [=](int l, int N) { return stoch_heat_eqn_energy_cc_l(l, N); }, 
        M, N, L, N0, Eps, output_convergence_filename, output_complexity_filename,
    output_regression_filename);
}

std::pair<std::vector<double>, std::vector<double>> stoch_heat_eqn_energy_cc_l(int l, int N) {
    const double lam = 0.25;
    const int batch_size = 100;
    int nf = 1 << (l + 1);
    double hf = 1.0 / nf;
    double dtf = lam * hf * hf;
    int timesteps_f = nf * nf;
    double std_f = std::sqrt(dtf / hf);

    std::vector<double> sum1(4, 0.0);
    std::vector<double> sum2(2, 0.0);

    for (int N1 = 0; N1 < N; N1 += batch_size) {
        int N2 = std::min(batch_size, N - N1);

        std::vector<double> Pf(N2, 0.0);
        std::vector<double> Pc(N2, 0.0);

        // Always construct fine grid
        std::vector<double> uf((nf + 1) * N2, 0.0);

        if (l == 0) {
            // Fine grid, no coarse grid
            std::vector<double> uf_new((nf + 1) * N2, 0.0); // temp buffer
            for (int t = 0; t < timesteps_f; ++t) {    // loop over the timesteps
                std::vector<double> dWf((nf - 1) * N2);
                for (int i = 0; i < (nf - 1) * N2; i++) // construct random noises for x point, for every trial
                    dWf[i] = std_f * Z(RNG);

                for (int i = 1; i < nf; i++) { // Loop over internal points
                    for (int n = 0; n < N2; ++n) { // loop over trials
                        uf_new[idx(i, n, N2)] = uf[idx(i, n, N2)] + lam * (uf[idx(i + 1, n, N2)] - 2 * uf[idx(i, n, N2)] + 
                        uf[idx(i - 1, n, N2)]) + dWf[idx(i - 1, n, N2)];
                    }
                }
                uf.swap(uf_new);
            }
            // Compute Pf
            for (int n = 0; n < N2; ++n) {
                double energy_sum = 0.0;
                for (int i = 0; i <= nf; i++) {
                    energy_sum += uf[idx(i, n, N2)] * uf[idx(i, n, N2)];
                }
                Pf[n] = hf * energy_sum;
            }
        } else {
            // Both fine and coarse grids need to be constructed
            int nc = nf / 2;
            double hc = 1.0 / nc;
            int timesteps_c = nc * nc;

            std::vector<double> uc((nc + 1) * N2);
            std::vector<double> uf_new((nf + 1) * N2, 0.0);
            std::vector<double> uc_new((nc + 1) * N2, 0.0);

            // half cell 
            int num_half_cells = 2 * (nf - 1);
            double std_half = std::sqrt(hf * dtf / 2.0);

            for (int tc = 0; tc < timesteps_c; ++tc) { // loop over coarse timesteps
                std::vector<double> dWc((nc - 1) * N2, 0.0);

                for (int s = 0; s < 4; ++s) { // loop through 4 fine timesteps per coarse timestep
                    
                    // 1. Generate half-cell noises
                    std::vector<double> half_cell_noises(num_half_cells * N2);
                    for(int i = 0; i < num_half_cells * N2; ++i) {
                        half_cell_noises[i] = std_half * Z(RNG);
                    }

                    // 2. Construct fine cell noises
                    std::vector<double> dWf((nf - 1) * N2);
                    for (int i = 0; i < nf - 1; ++i) {
                        for (int n = 0; n < N2; ++n) {
                            double zeta_L = half_cell_noises[idx(2 * i, n, N2)];
                            double zeta_R = half_cell_noises[idx(2 * i + 1, n, N2)];
                            dWf[idx(i, n, N2)] = (zeta_L + zeta_R) / hf;
                        }
                    }

                    // 3. Perform uf updates
                    for (int i = 1; i < nf; ++i) {// loop over internal fine points
                        for (int n = 0; n < N2; ++n) {// loop over trials
                            uf_new[idx(i, n, N2)] = uf[idx(i, n, N2)] + lam * (uf[idx(i + 1, n, N2)] - 2 * uf[idx(i, n, N2)] + uf[idx(i - 1, n, N2)])
                            + dWf[idx(i - 1, n, N2)];
                        }
                    }
                    uf.swap(uf_new);

                    // 4. Construct and accumulate coarse grid noise (dWc) from half-cells
                    for (int ic = 0; ic < nc - 1; ++ic) {
                        for (int n = 0; n < N2; ++n) {
                            // Sum the four half-cell noises corresponding to the coarse cell
                                double zeta_2k_L = half_cell_noises[idx(4 * ic, n, N2)];
                                double zeta_2k_R = half_cell_noises[idx(4 * ic + 1, n, N2)];
                                double zeta_2kp1_L = half_cell_noises[idx(4 * ic + 2, n, N2)];
                                double zeta_2kp1_R = half_cell_noises[idx(4 * ic + 3, n, N2)];
                                
                                dWc[idx(ic, n, N2)] += (zeta_2k_L + zeta_2k_R + zeta_2kp1_L + zeta_2kp1_R);
                        }
                    }
                }                
                
                // Coarse update (dWc scaled by 0.5)
                for (int ic = 1; ic < nc; ++ic)
                    for (int n = 0; n < N2; ++n)
                        uc_new[idx(ic, n, N2)] = uc[idx(ic, n, N2)] + 
                            lam * (uc[idx(ic + 1, n, N2)] -2 * uc[idx(ic, n, N2)] + uc[idx(ic - 1, n, N2)])
                            + (1.0 / hc) * dWc[idx(ic - 1, n, N2)];
                uc.swap(uc_new);
            }
            // Compute Pf and Pc
            for (int n =  0; n < N2; ++n) {
                double energy_sum_f = 0.0;
                double energy_sum_c = 0.0;
                for (int i_f = 0; i_f <= nf; ++i_f) {
                    energy_sum_f += uf[idx(i_f, n, N2)] * uf[idx(i_f, n, N2)];
                }
                for (int ic = 0; ic <= nc; ++ic) {
                    energy_sum_c += uc[idx(ic, n, N2)] * uc[idx(ic, n, N2)];
                }
                Pf[n] = hf * energy_sum_f;
                Pc[n] = hc * energy_sum_c;
            }
        }
        for (int n = 0; n < N2; ++n) {
            double diff = Pf[n] - Pc[n];
            sum1[0] += diff;
            sum1[1] += diff * diff;
            sum1[2] += diff * diff * diff;
            sum1[3] += diff * diff * diff * diff;
            sum2[0] += Pf[n];
            sum2[1] += Pf[n] * Pf[n];
        }
    }
    return {sum1, sum2};
}
