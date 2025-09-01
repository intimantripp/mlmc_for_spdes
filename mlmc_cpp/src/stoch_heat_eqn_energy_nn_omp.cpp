#include "stoch_heat_eqn_energy_nn.hpp"
#include "mlmc_test.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

// --- Reproducible per-(level,sample) seed ---
static inline uint64_t splitmix64(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

void run_stoch_heat_eqn_energy_nn(const int N) {
    std::cout << "Running MLMC Stochastic Heat Equation - Energy NN (OpenMP)\n" << std::endl;

    const int M  = 8;
    const int L  = 8;
    const int N0 = 100;
    const std::vector<double> Eps = {0.01, 0.05, 0.001, 0.005};

    const std::string output_convergence_filename = "../outputs_omp/mlmc_convergence_stoch_heat_eqn_energy_nn.csv";
    const std::string output_complexity_filename  = "../outputs_omp/mlmc_complexity_stoch_heat_eqn_energy_nn.csv";
    const std::string output_regression_filename  = "../outputs_omp/mlmc_regression_stoch_heat_eqn_energy_nn.csv";

    mlmc_test(
        [=](int l, int N) { return stoch_heat_eqn_energy_nn_l(l, N); },
        M, N, L, N0, Eps,
        output_convergence_filename,
        output_complexity_filename,
        output_regression_filename
    );
}

std::pair<std::vector<double>, std::vector<double>> stoch_heat_eqn_energy_nn_l(int l, int N) {
    const double lam = 0.25;

    // Fine grid
    const int    nf      = 1 << (l + 1);
    const double hf      = 1.0 / nf;
    const double dtf     = lam * hf * hf;
    const int    steps_f = nf * nf;
    const double std_f   = std::sqrt(dtf / hf);

    // Coarse grid (only used when l > 0)
    const int    nc      = (l == 0) ? 1 : nf / 2;
    const double hc      = (l == 0) ? 1.0 : 1.0 / nc;
    const int    steps_c = (l == 0) ? 0   : nc * nc;

    // MLMC moment sums
    double s10=0.0, s11=0.0, s12=0.0, s13=0.0; // E[Y], E[Y^2], E[Y^3], E[Y^4]
    double s20=0.0, s21=0.0;                   // E[Pf], E[Pf^2]

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:s10,s11,s12,s13,s20,s21)
    #endif
    for (int n = 0; n < N; ++n) {
        // Per-sample RNG (deterministic in l,n)
        std::mt19937_64 gen(splitmix64(0xCC00FFEEULL ^ (uint64_t(l) << 32) ^ (uint64_t)n));
        std::normal_distribution<> nd(0.0, 1.0);

        // Fine state
        std::vector<double> uf(nf + 1, 0.0), uf_new(nf + 1, 0.0);
        double Pf = 0.0, Pc = 0.0;

        if (l == 0) {
            // Level 0: iid nodal noise on fine grid
            std::vector<double> dWf(nf - 1);
            for (int t = 0; t < steps_f; ++t) {
                for (int i = 0; i < nf - 1; ++i) dWf[i] = std_f * nd(gen);
                for (int i = 1; i < nf; ++i) {
                    uf_new[i] = uf[i]
                              + lam * (uf[i + 1] - 2.0 * uf[i] + uf[i - 1])
                              + dWf[i - 1];
                }
                std::swap(uf, uf_new);
            }
            double sf = 0.0;
            for (int i = 0; i <= nf; ++i) sf += uf[i] * uf[i];
            Pf = hf * sf; // Y = Pf
        } else {
            // Levels l>0: NN coupling (coarse increments are sums of adjacent fine increments)
            std::vector<double> uc(nc + 1, 0.0), uc_new(nc + 1, 0.0);
            std::vector<double> dWf(nf - 1);
            std::vector<double> dWc(nc - 1, 0.0);

            for (int tc = 0; tc < steps_c; ++tc) {
                std::fill(dWc.begin(), dWc.end(), 0.0);

                // 4 fine substeps per coarse step
                for (int s = 0; s < 4; ++s) {
                    // Fine increments ~ N(0, std_f^2) independently per fine cell
                    for (int i = 0; i < nf - 1; ++i) dWf[i] = std_f * nd(gen);

                    // Fine update
                    for (int i = 1; i < nf; ++i) {
                        uf_new[i] = uf[i]
                                  + lam * (uf[i + 1] - 2.0 * uf[i] + uf[i - 1])
                                  + dWf[i - 1];
                    }
                    std::swap(uf, uf_new);

                    // NN coupling: accumulate coarse increments from adjacent fine
                    for (int ic = 0; ic < nc - 1; ++ic) {
                        dWc[ic] += dWf[2 * ic] + dWf[2 * ic + 1];
                    }
                }

                // Coarse update (same lam; injection factor 0.5 as in your serial code)
                for (int ic = 1; ic < nc; ++ic) {
                    uc_new[ic] = uc[ic]
                               + lam * (uc[ic + 1] - 2.0 * uc[ic] + uc[ic - 1])
                               + 0.5 * dWc[ic - 1];
                }
                std::swap(uc, uc_new);
            }

            // Energies
            double sf = 0.0, sc = 0.0;
            for (int i = 0; i <= nf; ++i) sf += uf[i] * uf[i];
            for (int i = 0; i <= nc; ++i) sc += uc[i] * uc[i];
            Pf = hf * sf;
            Pc = hc * sc; // Y = Pf - Pc
        }

        const double Y = (l == 0) ? Pf : (Pf - Pc);
        s10 += Y;
        s11 += Y * Y;
        s12 += Y * Y * Y;
        s13 += Y * Y * Y * Y;
        s20 += Pf;
        s21 += Pf * Pf;
    }

    std::vector<double> sum1{ s10, s11, s12, s13 };
    std::vector<double> sum2{ s20, s21 };
    return { sum1, sum2 };
}
