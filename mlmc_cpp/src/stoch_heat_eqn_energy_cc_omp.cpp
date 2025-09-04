#include "stoch_heat_eqn_energy_cc.hpp"
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

#ifndef M_PI
constexpr double M_PI = std::acos(-1.0);
#endif

static inline uint64_t splitmix64(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline int idx(int i, int n, int N2) { return i * N2 + n; }


void run_stoch_heat_eqn_energy_cc(const int N) {
    std::cout << "Running MLMC Stochastic Heat Equation - Energy CC (OpenMP)\n" << std::endl;


    int M = 8;
    int L = 8;
    int N0 = 100;
    std::vector<double> Eps = {0.01, 0.05, 0.001, 0.005};

    std::string output_complexity_filename = "../outputs_omp/mlmc_complexity_stoch_heat_eqn_energy_cc.csv";
    std::string output_convergence_filename = "../outputs_omp/mlmc_convergence_stoch_heat_eqn_energy_cc.csv";
    std::string output_regression_filename = "../outputs_omp/mlmc_regression_stoch_heat_eqn_energy_cc.csv";

    mlmc_test(
        [=](int l, int N) { return stoch_heat_eqn_energy_cc_l(l, N); },
        M, N, L, N0, Eps,
        output_convergence_filename,
        output_complexity_filename,
        output_regression_filename
    );
}


std::pair<std::vector<double>, std::vector<double>> stoch_heat_eqn_energy_cc_l(int l, int N) {
    const double lam = 0.25;

    // Fine grid
    const int nf = 1 << (l + 1);
    const double hf = 1.0 / nf;
    const double dtf = lam * hf * hf;
    const int steps_f = nf * nf;
    const double std_f = std::sqrt(dtf / hf);

    // Coarse grid
    const int nc = (l == 0) ? 1 : nf / 2;
    const double hc = (l == 0) ? 1.0 : 1.0 / nc;
    const int steps_c = (l == 0) ? 0 : nc * nc;

    // Half-cell setup
    const int num_half_cells = 2 * (nf - 1);
    const double std_half = std::sqrt(hf * dtf / 2.0);

    // Moment reductions: Y = Pf - Pc (or Pf on l=0)
    double s10=0.0, s11=0.0, s12=0.0, s13=0.0; // E[Y], E[Y^2], E[Y^3], E[Y^4]
    double s20=0.0, s21=0.0;                   // E[Pf], E[Pf^2]

    #ifdef _OPENMP 
    #pragma omp parallel for schedule(static) reduction(+:s10,s11,s12,s13,s20,s21)
    #endif
    for (int n = 0; n < N; ++n) {
    
        std::mt19937_64 gen(splitmix64(0xCC00FFEEULL ^ (uint64_t(l) << 32) ^ (uint64_t)n));
        std::normal_distribution<> nd(0.0, 1.0);

        // Fine 
        std::vector<double> uf(nf + 1, 0.0), uf_new(nf + 1, 0.0);
        double Pf = 0.0, Pc = 0.0;

        if (l == 0) {
            // Level 0: fine only
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

            double ef = 0.0;
            for (int i = 0; i <= nf; ++i) ef += uf[i] * uf[i];
            Pf = hf * ef;
        } else {
            // Coarse state
            std::vector<double> uc(nc + 1, 0.0), uc_new(nc + 1, 0.0);

            // Buffers
            std::vector<double> half_cell_noises(num_half_cells);
            std::vector<double> dWf(nf - 1);
            std::vector<double> dWc(nc - 1, 0.0);

            // Coarse loop
            for (int tc = 0; tc < steps_c; ++tc) {
                std::fill(dWc.begin(), dWc.end(), 0.0);

                // 4 fine substeps per coarse step
                for (int s = 0; s < 4; ++s) {
                    // 1) Generate half-cell noises ~ N(0, std_half^2)
                    for (int i = 0; i < num_half_cells; ++i) {
                        half_cell_noises[i] = std_half * nd(gen);
                    }

                    // 2) Construct fine cell noises: dWf[i] = (zeta_L + zeta_R)/hf
                    for (int i = 0; i < nf - 1; ++i) {
                        const double zL = half_cell_noises[2 * i];
                        const double zR = half_cell_noises[2 * i + 1];
                        dWf[i] = (zL + zR) / hf;
                    }

                    // 3) Fine update (interior nodes)
                    for (int i = 1; i < nf; ++i) {
                        uf_new[i] = uf[i]
                                  + lam * (uf[i + 1] - 2.0 * uf[i] + uf[i - 1])
                                  + dWf[i - 1];
                    }
                    std::swap(uf, uf_new);

                    // 4) Accumulate coarse noise from four adjacent half-cells per coarse cell
                    for (int ic = 0; ic < nc - 1; ++ic) {
                        const double z2kL = half_cell_noises[4 * ic];
                        const double z2kR = half_cell_noises[4 * ic + 1];
                        const double z2kp1L = half_cell_noises[4 * ic + 2];
                        const double z2kp1R = half_cell_noises[4 * ic + 3];
                        dWc[ic] += (z2kL + z2kR + z2kp1L + z2kp1R);
                    }
                }

                // 5) Coarse update: uc_new[ic] = uc[...] + lam*Lapl(uc) + (1/hc)*dWc[ic-1]
                for (int ic = 1; ic < nc; ++ic) {
                    uc_new[ic] = uc[ic]
                               + lam * (uc[ic + 1] - 2.0 * uc[ic] + uc[ic - 1])
                               + (1.0 / hc) * dWc[ic - 1];
                }
                std::swap(uc, uc_new);
            }

            // QoIs
            double ef = 0.0, ec = 0.0;
            for (int i = 0; i <= nf; ++i) ef += uf[i] * uf[i];
            for (int i = 0; i <= nc; ++i) ec += uc[i] * uc[i];
            Pf = hf * ef;
            Pc = hc * ec;
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
    return {sum1, sum2};
}
