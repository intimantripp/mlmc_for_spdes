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
    std::string output_rates_filename = "../outputs/mlmc_rates_dk_nn.csv";

    mlmc_test(
        [=] (int l, int N_samples) { return dean_kawasaki_eqn_nn_l(l, N_samples); },
        M, N, L, N0, Eps, output_convergence_filename, output_complexity_filename,
        output_rates_filename
    );
}


// The main level simulation function
std::pair<std::vector<double>, std::vector<double>>
dean_kawasaki_eqn_nn_l(int l, int N) {
    // Constants and Parameters
    const double Z_0 = 1.0 / 8.273782635069178;
    auto rho_0 = [&](double x) {
        double s = std::sin(x - M_PI/2.0);
        return Z_0 * (1.0 + std::exp(-(s*s) / 2.0) / std::sqrt(2.0 * M_PI));
    };
    auto phi_fn = [&](double x) { return std::sin(x); };
    const double N_particles = 2.0e6;
    const double inv_sqrtN   = 1.0 / std::sqrt(N_particles);
    const double lam = 0.25;
    const int    batch_size = 1000;

    // Grid Setup 
    int    nf  = 1 << (l + 2);
    double hf  = 2.0 * M_PI / nf;
    double dtf = lam * hf * hf;
    int    timesteps_f = nf * nf;
    const double std_f = std::sqrt(dtf / hf);
    const double inv_2hf = 1.0 / (2.0 * hf);

    std::vector<double> sum1(4, 0.0);
    std::vector<double> sum2(2, 0.0);

    for (int N1 = 0; N1 < N; N1 += batch_size) {
        int N2 = std::min(batch_size, N - N1);

        // Fine grid coordinates and phi (once per batch)
        std::vector<double> xf(nf), phi_vals_f(nf);
        for (int i = 0; i < nf; ++i) { xf[i] = i * hf; phi_vals_f[i] = phi_fn(xf[i]); }

        // Deterministic initial mean 
        std::vector<double> rho_bar_f(nf);
        for (int i = 0; i < nf; ++i) rho_bar_f[i] = rho_0(xf[i]);

        // State arrays
        std::vector<double> rho_f(nf * N2);
        for (int i = 0; i < nf; ++i)
            for (int n = 0; n < N2; ++n)
                rho_f[i * N2 + n] = rho_bar_f[i];

        // Preallocate temporaries (fine)
        std::vector<double> sqrt_rho_f(nf * N2);
        std::vector<double> dWf       (nf * N2);
        std::vector<double> flux_f    (nf * N2);

        // Left/right neighbor index lists
        std::vector<int> iL_f(nf), iR_f(nf);
        for (int i = 0; i < nf; ++i) {
            iL_f[i] = (i == 0 ? nf - 1 : i - 1);
            iR_f[i] = (i == nf - 1 ? 0 : i + 1);
        }

        std::vector<double> Pf(N2, 0.0), Pc(N2, 0.0);

        if (l == 0) {
            // Fine-only evolution
            for (int t = 0; t < timesteps_f; ++t) {
                // sqrt(rho), dW, flux
                for (int k = 0; k < nf * N2; ++k)
                    sqrt_rho_f[k] = std::sqrt(std::max(0.0, rho_f[k]));
                for (int k = 0; k < nf * N2; ++k)
                    dWf[k] = std_f * Z(RNG);
                for (int k = 0; k < nf * N2; ++k)
                    flux_f[k] = sqrt_rho_f[k] * dWf[k];

                // stencil update
                for (int i = 0; i < nf; ++i) {
                    int iL = iL_f[i], iR = iR_f[i];
                    int base  = i * N2;
                    int baseL = iL * N2;
                    int baseR = iR * N2;
                    for (int n = 0; n < N2; ++n) {
                        int k  = base  + n;
                        int kL = baseL + n;
                        int kR = baseR + n;

                        double divergence = (flux_f[kR] - flux_f[kL]) * inv_2hf;
                        double laplacian  = lam * (rho_f[kR] - 2.0 * rho_f[k] + rho_f[kL]) * 0.5;
                        rho_f[k] += laplacian + divergence * inv_sqrtN;
                    }
                }
            }

            // QoI on fine
            for (int n = 0; n < N2; ++n) {
                double inner = 0.0;
                for (int i = 0; i < nf; ++i) {
                    double dev = rho_f[i * N2 + n] - rho_bar_f[i];
                    inner += dev * phi_vals_f[i];
                }
                inner *= hf;
                Pf[n] = N_particles * inner * inner;
            }

        } else {
            // Coarse grid setup 
            int    nc  = nf / 2;
            double hc  = 2.0 * M_PI / nc;
            int    timesteps_c = nc * nc;
            const double inv_2hc = 1.0 / (2.0 * hc);

            std::vector<double> xc(nc), phi_vals_c(nc);
            for (int i = 0; i < nc; ++i) { xc[i] = i * hc; phi_vals_c[i] = phi_fn(xc[i]); }

            std::vector<double> rho_bar_c(nc);
            for (int i = 0; i < nc; ++i) rho_bar_c[i] = rho_0(xc[i]);

            std::vector<double> rho_c(nc * N2);
            for (int i = 0; i < nc; ++i)
                for (int n = 0; n < N2; ++n)
                    rho_c[i * N2 + n] = rho_bar_c[i];

            // Preallocate temporaries 
            std::vector<double> dWc       (nc * N2, 0.0);
            std::vector<double> sqrt_rho_c(nc * N2);
            std::vector<double> flux_c    (nc * N2);

            // Neighbor indices - coarse
            std::vector<int> iL_c(nc), iR_c(nc);
            for (int i = 0; i < nc; ++i) {
                iL_c[i] = (i == 0 ? nc - 1 : i - 1);
                iR_c[i] = (i == nc - 1 ? 0 : i + 1);
            }

            // Coupled evolution: per coarse step, do 4 fine substeps and aggregate dWc
            for (int t = 0; t < timesteps_c; ++t) {
                std::fill(dWc.begin(), dWc.end(), 0.0);

                for (int s = 0; s < 4; ++s) {
                    // fine substep
                    for (int k = 0; k < nf * N2; ++k)
                        sqrt_rho_f[k] = std::sqrt(std::max(0.0, rho_f[k]));
                    for (int k = 0; k < nf * N2; ++k)
                        dWf[k] = std_f * Z(RNG);
                    for (int k = 0; k < nf * N2; ++k)
                        flux_f[k] = sqrt_rho_f[k] * dWf[k];

                    for (int i = 0; i < nf; ++i) {
                        int iL = iL_f[i], iR = iR_f[i];
                        int base  = i * N2;
                        int baseL = iL * N2;
                        int baseR = iR * N2;
                        for (int n = 0; n < N2; ++n) {
                            int k  = base  + n;
                            int kL = baseL + n;
                            int kR = baseR + n;

                            double divergence = (flux_f[kR] - flux_f[kL]) * inv_2hf;
                            double laplacian  = lam * (rho_f[kR] - 2.0 * rho_f[k] + rho_f[kL]) * 0.5;
                            rho_f[k] += laplacian + divergence * inv_sqrtN;
                        }
                    }
                    // accumulate coarse noise from fine noise (NN coupling)
                    for (int i = 0; i < nc; ++i) {
                        int f0 = (2 * i) * N2;
                        int f1 = (2 * i + 1) * N2;
                        int c0 = i * N2;
                        for (int n = 0; n < N2; ++n) {
                            dWc[c0 + n] += dWf[f0 + n] + dWf[f1 + n];
                        }
                    }
                }

                // scale to match variance on coarse grid
                for (int k = 0; k < nc * N2; ++k) dWc[k] *= 0.5;

                // coarse update with aggregated noise
                for (int k = 0; k < nc * N2; ++k)
                    sqrt_rho_c[k] = std::sqrt(std::max(0.0, rho_c[k]));
                for (int k = 0; k < nc * N2; ++k)
                    flux_c[k] = sqrt_rho_c[k] * dWc[k];

                for (int i = 0; i < nc; ++i) {
                    int iL = iL_c[i], iR = iR_c[i];
                    int base  = i * N2;
                    int baseL = iL * N2;
                    int baseR = iR * N2;
                    for (int n = 0; n < N2; ++n) {
                        int k  = base  + n;
                        int kL = baseL + n;
                        int kR = baseR + n;

                        double divergence = (flux_c[kR] - flux_c[kL]) * inv_2hc;
                        double laplacian  = lam * (rho_c[kR] - 2.0 * rho_c[k] + rho_c[kL]) * 0.5;
                        rho_c[k] += laplacian + divergence * inv_sqrtN;
                    }
                }
            }

            // QoI on fine & coarse
            for (int n = 0; n < N2; ++n) {
                double inner_f = 0.0;
                for (int i = 0; i < nf; ++i) {
                    double dev = rho_f[i * N2 + n] - rho_bar_f[i];
                    inner_f += dev * phi_vals_f[i];
                }
                inner_f *= hf;
                Pf[n] = N_particles * inner_f * inner_f;
            }

            for (int n = 0; n < N2; ++n) {
                double inner_c = 0.0;
                for (int i = 0; i < nc; ++i) {
                    double dev = rho_c[i * N2 + n] - rho_bar_c[i];
                    inner_c += dev * phi_vals_c[i];
                }
                inner_c *= hc;
                Pc[n] = N_particles * inner_c * inner_c;
            }
        }

        // Accumulate MLMC moments
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
