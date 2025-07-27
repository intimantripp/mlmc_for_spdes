#include "stoch_heat_eqn_energy.hpp"
#include "mlmc_test.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

// Global RNG
static std::mt19937 RNG{ std::random_device{}() };
static std::normal_distribution<> Z(0.0, 1.0);

// Utility to index a flattened 2D array: xi(i, n) = i * N2 + n
inline int idx(int i, int n, int N2) {return i * N2 + n; }

void run_stoch_heat_eqn_energy(const int N) {
    std::cout << "Running MLMC Stochastic Heat Equation\n" << std::endl;
    int M = 8;
    int L = 6;
    int N0 = 100;
    std::vector<double> Eps = {0.005, 0.01, 0.02, 0.05, 0.1};

    std::string output_convergence_filename = "../outputs/mlmc_convergence_stoch_heat_eqn_energy.csv";
    std::string output_complexity_filename = "../outputs/mlmc_complexity_stoch_heat_eqn_energy.csv";

    mlmc_test(
        [=](int l, int N) { return stoch_heat_eqn_energy_l(l, N); }, 
        M, N, L, N0, Eps, output_convergence_filename, output_complexity_filename);
}

std::pair<std::vector<double>, std::vector<double>> stoch_heat_eqn_energy_l(int l, int N) {
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
                double s = 0.0;
                for (int i = 0; i <= nf; i++)
                    s += uf[idx(i, n, N2)] * uf[idx(i, n, N2)];
                Pf[n] = hf * s;
            }
        } else {
            // Both fine and coarse grids need to be constructed
            int nc = nf / 2;
            double hc = 1.0 / nc;
            double dtc = lam * hc * hc;
            int timesteps_c = nc * nc;

            std::vector<double> uc((nc + 1) * N2);
            std::vector<double> uf_new((nf + 1) * N2, 0.0);
            std::vector<double> uc_new((nc + 1) * N2, 0.0);
            for (int tc = 0; tc < timesteps_c; ++tc) { // loop over coarse timesteps
                std::vector<double> dWc((nc - 1) * N2);
                for (int s = 0; s < 4; ++s) { // loop through 4 fine timesteps per coarse timestep

                    // Construct dWf
                    std::vector<double> dWf((nf - 1) * N2);
                    for (int i = 0; i < (nf - 1) * N2; ++i)
                        dWf[i] = std_f * Z(RNG);

                    // Perform uf updates
                    for (int i = 1; i < nf; ++i) {// loop over internal fine points
                        for (int n = 0; n < N2; ++n) {// loop over trials
                            uf_new[idx(i, n, N2)] = uf[idx(i, n, N2)] + lam * (uf[idx(i + 1, n, N2)] - 2 * uf[idx(i, n, N2)] + uf[idx(i - 1, n, N2)])
                            + dWf[idx(i - 1, n, N2)];
                        }
                    }
                    uf.swap(uf_new);

                    // Coupling: sum adjacent fine increments for coarse
                    for (int ic = 0; ic < nc - 1; ++ic) // loop over internal coarse points
                        for (int n = 0; n < N2; ++n) // loop over numbers for each trial
                                // Each dWc internal ic point is constructed by summing two left most dWf points 
                                dWc[idx(ic, n, N2)] += dWf[idx(2 * ic, n, N2)] + dWf[idx(2 * ic + 1, n, N2)];
                }
                // Coarse update (dWc scaled by 0.5)
                for (int ic = 1; ic < nc; ++ic)
                    for (int n = 0; n < N2; ++n)
                        uc_new[idx(ic, n, N2)] = uc[idx(ic, n, N2)] + 
                            lam * (uc[idx(ic + 1, n, N2)] -2 * uc[idx(ic, n, N2)] + uc[idx(ic - 1, n, N2)])
                            + 0.5 * dWc[idx(ic - 1, n, N2)];
                uc.swap(uc_new);
            }
            // Compute Pf and Pc
            for (int n =  0; n < N2; ++n) {
                double sf = 0.0, sc = 0.0;
                for (int i_f = 0; i_f <= nf; ++i_f)
                    sf += uf[idx(i_f, n, N2)] * uf[idx(i_f, n, N2)]; // sum of values squared
                if (l > 0) {
                    for (int ic = 0; ic <= nc; ++ic)
                        sc += uc[idx(ic, n, N2)] * uc[idx(ic, n, N2)];
                }
                Pf[n] = hf * sf;
                Pc[n] = hc * sc;
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
            
        
        