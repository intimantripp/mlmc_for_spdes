#include "gbm_euler.hpp"
#include "mlmc_test.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm> // for std::max
#include <functional>
#include <string>

void run_gbm_euler(const int N) {
    std::cout << "Running GBM Euler Option Pricing MLMC..." <<std::endl;
    double S0 = 100.0;
    double K = 100.0;
    double  T = 1.0;
    double r = 0.05;
    double sig =  0.2;

    int M = 2;
    int L = 10;
    int N0 = 100;

    std::vector<double> Eps = {0.005, 0.01, 0.02, 0.05, 0.1};
    
    std::string output_convergence_filename = "../outputs/mlmc_convergence_gbm_euler.csv";
    std::string output_complexity_filename = ":../outputs/mlmc_complexity_gbm_euler.csv";

    mlmc_test([=](int l, int N) { return gbm_l(l, N, S0, K, T, r, sig); }, M, N, L, N0, Eps, 
    output_convergence_filename, output_complexity_filename);
    std::cout << "Finished running GBM Euler MLMC." << std::endl;
}

std::pair<std::vector<double>, std::vector<double>> gbm_l(
    int l, int N, double S0, double K, double T, double r, double sig
) {
    const int M = 2;
    int nf = std::pow(M, l);
    int nc = (l == 0) ? 1: nf / M;
    double hf = T / nf;
    double hc = (l == 0) ? T: T / nc;
    
    std::vector<double> sum1(4, 0.0);
    std::vector<double> sum2(2, 0.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);

    for (int N1 = 0; N1 < N; N1 += 10000) {
        int N2 = std::min(10000, N - N1);

        std::vector<double> Sf(N2, S0);
        std::vector<double> Sc = Sf;  // deep copy
        std::vector<double> Pc(N2, 0.0);
        std::vector<double> Pf(N2, 0.0);

        if (l==0) {
            // Fine path only
            for (int i = 0; i < N2; ++i) {
                double dWf = std::sqrt(hf) * d(gen);
                Sf[i] += r * Sf[i] * hf + sig * Sf[i] * dWf;
            }
        } else {
            // Multilevel paths
            for (int j = 0; j < nc; ++j) {
                std::vector<double> dWc(N2, 0.0);
                for (int k = 0; k < M; ++k) {
                    for (int i = 0; i < N2; ++i) {
                        double dWf = std::sqrt(hf) * d(gen);
                        dWc[i] += dWf;
                        Sf[i] += r * Sf[i] * hf + sig * Sf[i] * dWf;
                    }
                }
                for (int i = 0; i < N2; ++i) {
                    Sc[i] += r * Sc[i] * hc + sig * Sc[i] * dWc[i];
                }
            }
        }
        
        // European option payoff + discounting
        for (int i = 0; i < N2; ++i) {
            Pf[i] = std::exp(-r * T) * std::max(0.0, Sf[i] - K);
            if (l > 0) {
                Pc[i] = std::exp(-r * T) * std::max(0.0, Sc[i] - K);
            }
        }

        for (int i = 0; i < N2; ++i) {
            double diff = Pf[i] - Pc[i];
            sum1[0] += diff;
            sum1[1] += diff * diff;
            sum1[2] += diff * diff * diff;
            sum1[3] += diff * diff * diff * diff;
            sum2[0] += Pf[i];
            sum2[1] += Pf[i] * Pf[i];
        }
    }

    return {sum1, sum2};
}



