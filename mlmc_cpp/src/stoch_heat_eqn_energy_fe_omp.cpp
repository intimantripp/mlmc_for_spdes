#include "stoch_heat_eqn_energy_fe.hpp"
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

// Define pi if it isn't defined
#ifndef M_PI
constexpr double M_PI = std::acos(-1.0);
#endif

static inline uint64_t splitmix64(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Define Fourier Mode
static const int mode = 1;


// Utility to index a flattened 2D array: xi(i, n) = i * N2 + n
inline int idx(int i, int n, int N2) {return i * N2 + n; }

void run_stoch_heat_eqn_energy_fe(const int N) {
    std::cout << "Running MLMC Stochastic Heat Equation - Energy FE\n" << std::endl;
    int M = 8;
    int L = 5;
    int N0 = 100;
    std::vector<double> Eps = {0.00025, 0.0005, 0.001, 0.005, 0.01};

    std::string output_complexity_filename = "../outputs/mlmc_complexity_stoch_heat_eqn_energy_fe.csv";
    std::string output_convergence_filename = "../outputs/mlmc_convergence_stoch_heat_eqn_energy_fe.csv";
    std::string output_regression_filename = "../outputs/mlmc_regression_stoch_heat_eqn_energy_fe.csv";
    

    mlmc_test(
        [=](int l, int N) { return stoch_heat_eqn_energy_fe_l(l, N); }, 
        M, N, L, N0, Eps, output_convergence_filename, output_complexity_filename,
    output_regression_filename);
}

std::pair<std::vector<double>, std::vector<double>> stoch_heat_eqn_energy_fe_l(int l, int N) {
    
    const double lam = 0.25;
    
    // Fine grid quantities
    int nf = 1 << (l + 1);
    double hf = 1.0 / nf;
    double dtf = lam * hf * hf;
    int timesteps_f = nf * nf;
    double std_f = std::sqrt(dtf / hf);
    const double fac_f = std::sqrt(dtf) / hf;

    // Coarse grid quantities
    const int    nc = (l==0 ? 1 : nf/2);
    const double hc = (l==0 ? 1.0 : 1.0 / nc);
    const double dtc = lam * hc * hc;
    const int    steps_c = (l==0 ? 0 : nc * nc);
    const double fac_c = (l==0 ? 0.0 : std::sqrt(dtc) / hc);

    // Moments
    double s10=0.0, s11=0.0, s12=0.0, s13=0.0; // moments of Y
    double s20=0.0, s21=0.0;                   // moments of Pf

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:s10, s11, s12, s13, s20, s21)
    #endif
    for (int n = 0; n < N; ++n) {

        std::mt19937_64 gen(splitmix64(0xC0FFEEULL ^ (uint64_t(l)<<32) ^ (uint64_t)n));
        std::normal_distribution<> nd(0.0, 1.0);

        // Always construct fine grid
        std::vector<double> uf(nf+1), uf_new(nf+1, 0.0);
        double Pf = 0.0, Pc = 0.0;

        if (l == 0) {
            // Fine grid. Generate correlated noise and update the fine grid
            std::vector<double> Z_node(nf-1), Z_edge(nf), dWf(nf-1);
            for (int t = 0; t < timesteps_f; ++t) {

                // 1) generate base noises
                for (int i = 0; i < nf-1; ++i) Z_node[i] = nd(gen);
                for (int i = 0; i < nf;   ++i) Z_edge[i] = nd(gen);

                // 2) construct correlated nodal noise increments on fine grid
                //    dWf[i] = sqrt(hf/3)*Z_node[i] + sqrt(hf/6)*(Z_edge[i] + Z_edge[i+1])
                for (int i = 0; i < nf-1; ++i) {
                    dWf[i] = std::sqrt(hf/3.0)*Z_node[i]
                           + std::sqrt(hf/6.0)*(Z_edge[i] + Z_edge[i+1]);
                }

                // 3) FE update on interior points
                for (int i = 1; i < nf; ++i) {
                    uf_new[i] = uf[i]
                              + lam * (uf[i+1] - 2.0*uf[i] + uf[i-1])
                              + fac_f * dWf[i-1];
                }

                std::swap(uf, uf_new);
            }

            // energy QoI on fine grid
            double ef = 0.0;
            for (int i = 0; i <= nf; ++i) ef += uf[i]*uf[i];
            Pf = hf * ef;
            // level 0 has no coarse, so Y = Pf and Pc = 0
        } 
        
        else {

            // coarse grid
            std::vector<double> uc(nc+1, 0.0), uc_new(nc+1, 0.0);
            
            // noise arrays
            std::vector<double> Z_node(nf-1), Z_edge(nf), dWf(nf-1);
            std::vector<double> dWc(nc-1, 0.0);

            for (int tc = 0; tc < steps_c; ++tc) {
                // Coarse noises getting reset here
                std::fill(dWc.begin(), dWc.end(), 0.0);

                // doing my 4 fine substeps per coarse step
                for (int s = 0; s < 4; ++s) {
                    // 1) base noises on the fine grid
                    for (int i = 0; i < nf - 1; ++i) Z_node[i] = nd(gen);
                    for (int i = 0; i < nf;   ++i) Z_edge[i] = nd(gen);

                    // 2) correlate my fine increments
                    for (int i = 0; i < nf-1; ++i) {
                        dWf[i] = std::sqrt(hf/3.0)*Z_node[i]
                               + std::sqrt(hf/6.0)*(Z_edge[i] + Z_edge[i+1]);
                    }

                    // 3) update uf
                    for (int i = 1; i < nf; ++i) {
                        uf_new[i] = uf[i] + lam * (uf[i+1] - 2.0*uf[i] + uf[i-1])
                                    + fac_f * dWf[i-1];
                    }
                    std::swap(uf, uf_new);

                    // 4) accumulate my fine noises into the coarse ones
                    for (int ic = 0; ic < nc-1; ++ic) {
                        dWc[ic] += 0.5 * dWf[2*ic] + dWf[2*ic + 1] + 0.5 * dWf[2*ic + 2];
                    }
                }

                // 5) update the coarse grid now
                for (int ic = 1; ic < nc; ++ic) {
                    uc_new[ic] = uc[ic] 
                                 + lam * (uc[ic+1] - 2.0*uc[ic] + uc[ic-1])
                                 + fac_c * (0.5 * dWc[ic-1]);
                }
                std::swap(uc, uc_new);
            }

            // QoIs again
            double ef = 0.0, ec = 0.0;
            for (int i = 0; i <= nf; ++i) ef += uf[i] * uf[i];
            for (int i = 0; i <= nc; ++i) ec += uc[i] * uc[i];
            Pf = hf * ef;
            Pc = hc * ec;
        }

        const double Y = (l==0) ? Pf : (Pf - Pc);
        s10 += Y;
        s11 += Y*Y;
        s12 += Y*Y*Y;
        s13 += Y*Y*Y*Y;
        s20 += Pf;
        s21 += Pf*Pf;
    }
    std::vector<double> sum1{ s10, s11, s12, s13 };
    std::vector<double> sum2{ s20, s21 };
    return {sum1, sum2};

}
