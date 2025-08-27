// Thisis my OpenMP varince of the GBM Euler MLM implementation. Going to first build it to verify it's all 
// working.
#include "gbm_euler.hpp"
#include "mlmc_test.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <array>
#include <random>
#include <algorithm>
#include <functional>
#include <string>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

// tiny 64-bit mixer for deterministic seeding of threads - very clever thing
static inline uint64_t splitmix64(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

void run_gbm_euler(const int N) {
    std::cout << "Running GBM Euler Option Pricing MLMC with OMP ..." <<std::endl;
    double S0 = 100.0;
    double K = 100.0;
    double  T = 1.0;
    double r = 0.05;
    double sig =  0.2;

    int M = 2;
    int L = 10;
    int N0 = 100;

    std::vector<double> Eps = {0.005, 0.01, 0.02, 0.05, 0.1};
    
    std::string output_convergence_filename = "outputs_omp/mlmc_convergence_gbm_euler.csv";
    std::string output_complexity_filename = "outputs_omp/mlmc_complexity_gbm_euler.csv";
    std::string output_regression_filename = "outputs_omp/mlmc_regression_gbm_euler.csv";

    mlmc_test([=](int l, int N) { return gbm_l(l, N, S0, K, T, r, sig); }, M, N, L, N0, Eps, 
    output_convergence_filename, output_complexity_filename, output_regression_filename);
    std::cout << "Finished running GBM Euler OMP MLMC." << std::endl;
}

#include <cstdint>  // for uint64_t

std::pair<std::vector<double>, std::vector<double>>
gbm_l(int l, int N, double S0, double K, double T, double r, double sig) {
    const int  M  = 2;
    const int  nf = 1 << l;  
    const int  nc = (l == 0) ? 1 : nf / M;
    const double hf = T / nf;
    const double hc = (l == 0) ? T : T / nc;
    const double discount = std::exp(-r * T);
    const double sqrt_hf  = std::sqrt(hf);

    std::vector<double> sum1(4, 0.0), sum2(2, 0.0);
    double s10=0.0, s11=0.0, s12=0.0, s13=0.0;
    double s20=0.0, s21=0.0;

    // If less than 5000 samples, don't use OpenMP
    constexpr int PAR_THRESHOLD = 1000;
    if (N < PAR_THRESHOLD) {
        for (int i = 0; i < N; ++i) {
            std::mt19937_64 gen(splitmix64(0xC0FFEEULL ^ (uint64_t(l) << 32) ^ (uint64_t)i));
            std::normal_distribution<> nd(0.0, 1.0);

            double Sf = S0, Sc = S0;
            if (l == 0) {
                const double dWf = sqrt_hf * nd(gen);
                Sf += r*Sf*hf + sig*Sf*dWf;
            } else {
                for (int j = 0; j < nc; ++j) {
                    double dWc = 0.0;
                    for (int k = 0; k < M; ++k) {
                        const double dWf = sqrt_hf * nd(gen);
                        dWc += dWf;
                        Sf  += r*Sf*hf + sig*Sf*dWf;
                    }
                    Sc += r*Sc*hc + sig*Sc*dWc;
                }
            }
            const double Pf = discount * std::max(0.0, Sf - K);
            const double Pc = (l > 0) ? discount * std::max(0.0, Sc - K) : 0.0;
            const double Y  = Pf - Pc;

            s10 += Y;              s11 += Y*Y;
            s12 += Y*Y*Y;          s13 += Y*Y*Y*Y;
            s20 += Pf;             s21 += Pf*Pf;
        }
    } else {
        // Thread over samples
        #pragma omp parallel for if(!omp_in_parallel()) schedule(static) \
                                 reduction(+:s10,s11,s12,s13,s20,s21)
        for (int i = 0; i < N; ++i) {
            std::mt19937_64 gen(splitmix64(0xC0FFEEULL ^ (uint64_t(l) << 32) ^ (uint64_t)i));
            std::normal_distribution<> nd(0.0, 1.0);

            double Sf = S0, Sc = S0;
            if (l == 0) {
                const double dWf = sqrt_hf * nd(gen);
                Sf += r*Sf*hf + sig*Sf*dWf;
            } else {
                for (int j = 0; j < nc; ++j) {
                    double dWc = 0.0;
                    for (int k = 0; k < M; ++k) {
                        const double dWf = sqrt_hf * nd(gen);
                        dWc += dWf;
                        Sf  += r*Sf*hf + sig*Sf*dWf;
                    }
                    Sc += r*Sc*hc + sig*Sc*dWc;
                }
            }
            const double Pf = discount * std::max(0.0, Sf - K);
            const double Pc = (l > 0) ? discount * std::max(0.0, Sc - K) : 0.0;
            const double Y  = Pf - Pc;

            s10 += Y;              s11 += Y*Y;
            s12 += Y*Y*Y;          s13 += Y*Y*Y*Y;
            s20 += Pf;             s21 += Pf*Pf;
        }
    }

    // write totals
    sum1[0]=s10; sum1[1]=s11; sum1[2]=s12; sum1[3]=s13;
    sum2[0]=s20; sum2[1]=s21;
    return {sum1, sum2};
}
