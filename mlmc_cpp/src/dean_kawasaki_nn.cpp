#include "mlmc_test.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <functional>
#include <random>

// --- Assumed to be available from your existing code ---
// Global RNG
static std::mt19937 RNG{ std::random_device{}() };
static std::normal_distribution<> Z(0.0, 1.0);


inline int idx(int i, int n, int N_trials) { return i * N_trials + n; }

void roll(std::vector<double>& vec, int shift, int n_points, int N2) {
    if (shift == 0) return;
    std::vector<double> temp = vec;
    for (int n = 0; n < N2; ++n) {
        for (int i = 0; i < n_points; ++i) {
            int new_i = (i - shift + n_points) % n_points;
            vec[idx(i, n, N2)] = temp[idx(new_i, n, N2)];
        }
    }
}

std::pair<std::vector<double>, std::vector<double>> dean_kawasaki_eqn_nn_l(int l, int N);


void run_dean_kawasaki_nn(const int N) {
    std::cout << "Running MLMC Dean-Kawasaki - NN\n" << std::endl;
    int M = 8;
    int L = 5;
    int N0 = 100;
    std::vector<double> Eps = {0.001, 0.005, 0.01, 0.05};

    std::string output_convergence_filename = "../outputs/mlmc_convergence_dk_nn.csv";
    std::string output_complexity_filename = "../outputs/mlmc_complexity_dk_nn.csv";

    mlmc_test(
        [=] (int l, int N_samples) { return dean_kawasaki_eqn_nn_l(l, N_samples); },
        M, N, L, N0, Eps, output_convergence_filename, output_complexity_filename
    );
}

// The main level simulation function
std::pair<std::vector<double>, std::vector<double>> dean_kawasaki_eqn_nn_l(int l, int N) {
    // --- Constants and Parameters ---
    const double Z_0 = 1.0 / 8.273782635069178;
    auto rho_0 = [&](double x) { return Z_0 * (1.0 + std::exp(-(std::sin(x - M_PI/2.0) * std::sin(x - M_PI/2.0)) / 2.0) / std::sqrt(2.0 * M_PI)); };
    auto phi_fn = [&](double x) { return std::sin(x); };
    const double N_particles = 2.0e6;
    const double lam = 0.25;
    const int batch_size = 1000;

    // --- Grid Setup ---
    int nf = 1 << (l + 2);
    double hf = 2.0 * M_PI / nf;
    double dtf = lam * hf * hf;
    int timesteps_f = nf * nf;

    std::vector<double> sum1(4, 0.0);
    std::vector<double> sum2(2, 0.0);

    for (int N1 = 0; N1 < N; N1 += batch_size) {
        int N2 = std::min(batch_size, N - N1);

        std::vector<double> xf(nf);
        for(int i=0; i<nf; ++i) xf[i] = i * hf;

        std::vector<double> rho_bar_f(nf);
        for(int i=0; i<nf; ++i) rho_bar_f[i] = rho_0(xf[i]);

        std::vector<double> rho_f(nf * N2);
        for (int i = 0; i < nf; ++i) {
            for (int n = 0; n < N2; ++n) {
                rho_f[idx(i, n, N2)] = rho_bar_f[i];
            }
        }

        std::vector<double> Pf(N2, 0.0);
        std::vector<double> Pc(N2, 0.0);

        if (l == 0) {
            for (int t = 0; t < timesteps_f; ++t) {
                std::vector<double> sqrt_rho(nf * N2);
                for(int i=0; i<nf*N2; ++i) sqrt_rho[i] = std::sqrt(std::max(0.0, rho_f[i]));
                
                std::vector<double> dW(nf * N2);
                double std_f = std::sqrt(dtf / hf);
                for(int i=0; i<nf*N2; ++i) dW[i] = std_f * Z(RNG);

                std::vector<double> flux(nf * N2);
                for(int i=0; i<nf*N2; ++i) flux[i] = sqrt_rho[i] * dW[i];
                
                std::vector<double> flux_p1 = flux, flux_m1 = flux;
                roll(flux_p1, -1, nf, N2);
                roll(flux_m1, 1, nf, N2);

                std::vector<double> rho_p1 = rho_f, rho_m1 = rho_f;
                roll(rho_p1, -1, nf, N2);
                roll(rho_m1, 1, nf, N2);
                
                for (int i = 0; i < nf * N2; ++i) {
                    double divergence = (flux_p1[i] - flux_m1[i]) / (2.0 * hf);
                    double laplacian = lam * (rho_p1[i] - 2.0 * rho_f[i] + rho_m1[i]) / 2.0;
                    rho_f[i] += laplacian + divergence / std::sqrt(N_particles);
                }
            }
            
            std::vector<double> phi_vals_f(nf);
            for(int i=0; i<nf; ++i) phi_vals_f[i] = phi_fn(xf[i]);

            for (int n = 0; n < N2; ++n) {
                double inner_product = 0.0;
                for (int i = 0; i < nf; ++i) {
                    double deviation = rho_f[idx(i, n, N2)] - rho_bar_f[i];
                    inner_product += deviation * phi_vals_f[i];
                }
                inner_product *= hf;
                Pf[n] = N_particles * inner_product * inner_product;
            }

        } else {
            int nc = nf / 2;
            double hc = 2.0 * M_PI / nc;
            int timesteps_c = nc * nc;

            std::vector<double> xc(nc);
            for(int i=0; i<nc; ++i) xc[i] = i * hc;

            std::vector<double> rho_bar_c(nc);
            for(int i=0; i<nc; ++i) rho_bar_c[i] = rho_0(xc[i]);
            
            std::vector<double> rho_c(nc * N2);
            for (int i = 0; i < nc; ++i) {
                for (int n = 0; n < N2; ++n) {
                    rho_c[idx(i, n, N2)] = rho_bar_c[i];
                }
            }

            for (int t = 0; t < timesteps_c; ++t) {
                std::vector<double> dWc(nc * N2, 0.0);
                for (int s = 0; s < 4; ++s) {
                    std::vector<double> sqrt_rho_f(nf * N2);
                    for(int i=0; i<nf*N2; ++i) sqrt_rho_f[i] = std::sqrt(std::max(0.0, rho_f[i]));
                    
                    std::vector<double> dWf(nf * N2);
                    double std_f = std::sqrt(dtf / hf);
                    for(int i=0; i<nf*N2; ++i) dWf[i] = std_f * Z(RNG);

                    std::vector<double> flux_f(nf*N2);
                    for(int i=0; i<nf*N2; ++i) flux_f[i] = sqrt_rho_f[i] * dWf[i];
                    
                    std::vector<double> flux_f_p1 = flux_f, flux_f_m1 = flux_f;
                    roll(flux_f_p1, -1, nf, N2);
                    roll(flux_f_m1, 1, nf, N2);

                    std::vector<double> rho_f_p1 = rho_f, rho_f_m1 = rho_f;
                    roll(rho_f_p1, -1, nf, N2);
                    roll(rho_f_m1, 1, nf, N2);

                    for (int i = 0; i < nf*N2; ++i) {
                         double divergence_f = (flux_f_p1[i] - flux_f_m1[i]) / (2.0 * hf);
                         double laplacian_f = lam * (rho_f_p1[i] - 2.0*rho_f[i] + rho_f_m1[i]) / 2.0;
                         rho_f[i] += laplacian_f + divergence_f / std::sqrt(N_particles);
                    }
                    
                    for (int i = 0; i < nc; ++i) {
                        for (int n = 0; n < N2; ++n) {
                            dWc[idx(i, n, N2)] += dWf[idx(2 * i, n, N2)] + dWf[idx(2 * i + 1, n, N2)];
                        }
                    }
                }
                
                for(int i=0; i<nc*N2; ++i) dWc[i] *= 0.5;

                std::vector<double> sqrt_rho_c(nc*N2);
                for(int i=0; i<nc*N2; ++i) sqrt_rho_c[i] = std::sqrt(std::max(0.0, rho_c[i]));

                std::vector<double> flux_c(nc*N2);
                for(int i=0; i<nc*N2; ++i) flux_c[i] = sqrt_rho_c[i] * dWc[i];

                std::vector<double> flux_c_p1 = flux_c, flux_c_m1 = flux_c;
                roll(flux_c_p1, -1, nc, N2);
                roll(flux_c_m1, 1, nc, N2);

                std::vector<double> rho_c_p1 = rho_c, rho_c_m1 = rho_c;
                roll(rho_c_p1, -1, nc, N2);
                roll(rho_c_m1, 1, nc, N2);

                for (int i = 0; i < nc*N2; ++i) {
                    double divergence_c = (flux_c_p1[i] - flux_c_m1[i]) / (2.0 * hc);
                    double laplacian_c = lam * (rho_c_p1[i] - 2.0*rho_c[i] + rho_c_m1[i]) / 2.0;
                    rho_c[i] += laplacian_c + divergence_c / std::sqrt(N_particles);
                }
            }
            
            std::vector<double> phi_vals_f(nf);
            for(int i=0; i<nf; ++i) phi_vals_f[i] = phi_fn(xf[i]);
            
            for (int n = 0; n < N2; ++n) {
                double inner_product_f = 0.0;
                for (int i = 0; i < nf; ++i) {
                    double deviation_f = rho_f[idx(i, n, N2)] - rho_bar_f[i];
                    inner_product_f += deviation_f * phi_vals_f[i];
                }
                inner_product_f *= hf;
                Pf[n] = N_particles * inner_product_f * inner_product_f;
            }

            std::vector<double> phi_vals_c(nc);
            for(int i=0; i<nc; ++i) phi_vals_c[i] = phi_fn(xc[i]);

            for (int n = 0; n < N2; ++n) {
                double inner_product_c = 0.0;
                for (int i = 0; i < nc; ++i) {
                    double deviation_c = rho_c[idx(i, n, N2)] - rho_bar_c[i];
                    inner_product_c += deviation_c * phi_vals_c[i];
                }
                inner_product_c *= hc;
                Pc[n] = N_particles * inner_product_c * inner_product_c;
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
