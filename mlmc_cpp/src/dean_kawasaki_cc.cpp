#include "mlmc_test.hpp"
#include "dean_kawasaki_cc.hpp"
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

inline void roll(std::vector<double>& vec, int shift, int n_points, int N2) {
    if (shift == 0) return;
    std::vector<double> temp = vec;
    for (int n = 0; n < N2; ++n) {
        for (int i = 0; i < n_points; ++i) {
            int new_i = (i - shift + n_points) % n_points;
            vec[idx(i, n, N2)] = temp[idx(new_i, n, N2)];
        }
    }
}

void run_dean_kawasaki_cc(const int N) {
    std::cout << "Running MLMC Dean-Kawasaki - cc\n" << std::endl;
    int M = 8;
    int L = 5;
    int N0 = 100;
    std::vector<double> Eps = {0.001, 0.005, 0.01, 0.05};

    std::string output_convergence_filename = "../outputs/mlmc_convergence_dk_cc.csv";
    std::string output_complexity_filename = "../outputs/mlmc_complexity_dk_cc.csv";

    mlmc_test(
        [=] (int l, int N_samples) { return dean_kawasaki_eqn_cc_l(l, N_samples); },
        M, N, L, N0, Eps, output_convergence_filename, output_complexity_filename
    );
}

std::pair<std::vector<double>, std::vector<double>>
dean_kawasaki_eqn_cc_l(int l, int N)
{
    // --- Model/observable setup (unchanged from your nn version) ---
    const double Z_0 = 1.0 / 8.273782635069178;
    auto rho_0 = [&](double x) {
        return Z_0 * (1.0 + std::exp(-(std::sin(x - M_PI/2.0) * std::sin(x - M_PI/2.0)) / 2.0)
                           / std::sqrt(2.0 * M_PI));
    };
    auto phi_fn = [&](double x) { return std::sin(x); };

    const double N_particles = 2.0e6;  // controls stochastic intensity
    const double lam = 0.25;           // dt = lam * h^2 (explicit Euler)
    const int    batch_size = 1000;

    // --- Fine grid/time ---
    int    nf        = 1 << (l + 2);
    double hf        = 2.0 * M_PI / nf;
    double dtf       = lam * hf * hf;      // explicit Euler CFL form
    int    steps_f   = nf * nf;            // same total time as your nn code

    // Return containers: moments of Pf-Pc (sum1) and Pf alone (sum2)
    std::vector<double> sum1(4, 0.0);
    std::vector<double> sum2(2, 0.0);

    for (int N1 = 0; N1 < N; N1 += batch_size) {
        int N2 = std::min(batch_size, N - N1);

        // Fine grid x and mean profile
        std::vector<double> xf(nf);
        for (int i = 0; i < nf; ++i) xf[i] = i * hf;

        std::vector<double> rho_bar_f(nf);
        for (int i = 0; i < nf; ++i) rho_bar_f[i] = rho_0(xf[i]);

        // Initial fine state replicated across N2 samples
        std::vector<double> rho_f(nf * N2);
        for (int i = 0; i < nf; ++i)
            for (int n = 0; n < N2; ++n)
                rho_f[idx(i, n, N2)] = rho_bar_f[i];

        std::vector<double> Pf(N2, 0.0);
        std::vector<double> Pc(N2, 0.0);  // left as 0 at l=0, as per MLMC convention

        if (l == 0) {
            // ----------------------- Base level: standard evolution -----------------------
            for (int t = 0; t < steps_f; ++t) {
                // multiplicative noise: sqrt(rho)
                std::vector<double> sqrt_rho(nf * N2);
                for (int i = 0; i < nf * N2; ++i)
                    sqrt_rho[i] = std::sqrt(std::max(0.0, rho_f[i]));

                // Build dW_f using half-cell construction (not necessary at l=0,
                // but harmless and keeps variance scaling identical everywhere)
                const int    num_half_cells = 2 * nf;
                const double std_half       = std::sqrt(dtf / (2.0 * hf));
                std::vector<double> half_noise(num_half_cells * N2);
                for (int i = 0; i < num_half_cells * N2; ++i)
                    half_noise[i] = std_half * Z(RNG);

                std::vector<double> dWf(nf * N2);
                for (int i = 0; i < nf; ++i)
                    for (int n = 0; n < N2; ++n)
                        dWf[idx(i, n, N2)] =
                            half_noise[idx(2 * i,     n, N2)] +
                            half_noise[idx(2 * i + 1, n, N2)];  // Var = dtf/hf

                // flux = sqrt(rho) * dW
                std::vector<double> flux(nf * N2);
                for (int i = 0; i < nf * N2; ++i) flux[i] = sqrt_rho[i] * dWf[i];

                // centered divergence (cell-centered flux)
                std::vector<double> flux_p1 = flux, flux_m1 = flux;
                roll(flux_p1, -1, nf, N2);
                roll(flux_m1,  1, nf, N2);

                // laplacian term (explicit Euler): lam * (u_{i+1} - 2u_i + u_{i-1})
                std::vector<double> rho_p1 = rho_f, rho_m1 = rho_f;
                roll(rho_p1, -1, nf, N2);
                roll(rho_m1,  1, nf, N2);

                for (int i = 0; i < nf * N2; ++i) {
                    double divergence = (flux_p1[i] - flux_m1[i]) / (2.0 * hf);
                    double laplacian  = lam * (rho_p1[i] - 2.0 * rho_f[i] + rho_m1[i]) / 2;
                    rho_f[i] += laplacian + divergence / std::sqrt(N_particles);
                    // Optional: enforce tiny floor to help stability
                    // rho_f[i] = std::max(rho_f[i], 1e-14);
                }
            }

            // QoI on fine
            std::vector<double> phi_vals_f(nf);
            for (int i = 0; i < nf; ++i) phi_vals_f[i] = phi_fn(xf[i]);

            for (int n = 0; n < N2; ++n) {
                double proj = 0.0;
                for (int i = 0; i < nf; ++i) {
                    double dev = rho_f[idx(i, n, N2)] - rho_bar_f[i];
                    proj += dev * phi_vals_f[i];
                }
                proj *= hf;
                Pf[n] = N_particles * proj * proj;
            }
        } else {
            // ----------------------- MLMC level: half-cell coupling -----------------------
            int    nc       = nf / 2;
            double hc       = 2.0 * M_PI / nc;
            int    steps_c  = nc * nc;       // so that steps_c * dtc = steps_f * dtf
            double dtc      = 4.0 * dtf;     // because hc = 2*hf

            // coarse grid & mean
            std::vector<double> xc(nc);
            for (int i = 0; i < nc; ++i) xc[i] = i * hc;

            std::vector<double> rho_bar_c(nc);
            for (int i = 0; i < nc; ++i) rho_bar_c[i] = rho_0(xc[i]);

            std::vector<double> rho_c(nc * N2);
            for (int i = 0; i < nc; ++i)
                for (int n = 0; n < N2; ++n)
                    rho_c[idx(i, n, N2)] = rho_bar_c[i];

            // Half-cell parameters
            const int    num_half_cells = 2 * nf;
            const double std_half       = std::sqrt(dtf / (2.0 * hf));

            for (int t = 0; t < steps_c; ++t) {
                // Accumulate one coarse increment from 4 fine substeps
                std::vector<double> dWc(nc * N2, 0.0);

                for (int s = 0; s < 4; ++s) {
                    // ---- Fine substep s at fine dtf ----
                    // Build half-cell noises
                    std::vector<double> half_noise(num_half_cells * N2);
                    for (int i = 0; i < num_half_cells * N2; ++i)
                        half_noise[i] = std_half * Z(RNG);

                    // dW_f from half-cell sum
                    std::vector<double> dWf(nf * N2);
                    for (int i = 0; i < nf; ++i)
                        for (int n = 0; n < N2; ++n)
                            dWf[idx(i, n, N2)] =
                                half_noise[idx(2 * i,     n, N2)] +
                                half_noise[idx(2 * i + 1, n, N2)]; // Var = dtf/hf

                    // Fine update
                    std::vector<double> sqrt_rho_f(nf * N2);
                    for (int i = 0; i < nf * N2; ++i)
                        sqrt_rho_f[i] = std::sqrt(std::max(0.0, rho_f[i]));

                    std::vector<double> flux_f(nf * N2);
                    for (int i = 0; i < nf * N2; ++i) flux_f[i] = sqrt_rho_f[i] * dWf[i];

                    std::vector<double> flux_f_p1 = flux_f, flux_f_m1 = flux_f;
                    roll(flux_f_p1, -1, nf, N2);
                    roll(flux_f_m1,  1, nf, N2);

                    std::vector<double> rho_f_p1 = rho_f, rho_f_m1 = rho_f;
                    roll(rho_f_p1, -1, nf, N2);
                    roll(rho_f_m1,  1, nf, N2);

                    for (int i = 0; i < nf * N2; ++i) {
                        double divergence_f = (flux_f_p1[i] - flux_f_m1[i]) / (2.0 * hf);
                        double laplacian_f  = lam * (rho_f_p1[i] - 2.0 * rho_f[i] + rho_f_m1[i]) / 2.0;
                        rho_f[i] += laplacian_f + divergence_f / std::sqrt(N_particles);
                        // rho_f[i] = std::max(rho_f[i], 1e-14);
                    }

                    // ---- Build matched coarse increment from half-cells used above ----
                    // Each coarse cell corresponds to two fine cells: (2i, 2i+1).
                    // Over 4 fine substeps and 4 spatial half-cells per coarse cell,
                    // the variance sums to 8*dtf/hf; multiply by 0.5 to target dtc/hc = 2*dtf/hf.
                    for (int i = 0; i < nc; ++i) {
                        for (int n = 0; n < N2; ++n) {
                            double accum =
                                half_noise[idx(4 * i,     n, N2)] +
                                half_noise[idx(4 * i + 1, n, N2)] +
                                half_noise[idx(4 * i + 2, n, N2)] +
                                half_noise[idx(4 * i + 3, n, N2)];
                            dWc[idx(i, n, N2)] += 0.5 * accum;  // <-- critical scaling
                        }
                    }
                }

                // One coarse update using dWc
                std::vector<double> sqrt_rho_c(nc * N2);
                for (int i = 0; i < nc * N2; ++i)
                    sqrt_rho_c[i] = std::sqrt(std::max(0.0, rho_c[i]));

                std::vector<double> flux_c(nc * N2);
                for (int i = 0; i < nc * N2; ++i) flux_c[i] = sqrt_rho_c[i] * dWc[i];

                std::vector<double> flux_c_p1 = flux_c, flux_c_m1 = flux_c;
                roll(flux_c_p1, -1, nc, N2);
                roll(flux_c_m1,  1, nc, N2);

                std::vector<double> rho_c_p1 = rho_c, rho_c_m1 = rho_c;
                roll(rho_c_p1, -1, nc, N2);
                roll(rho_c_m1,  1, nc, N2);

                for (int i = 0; i < nc * N2; ++i) {
                    double divergence_c = (flux_c_p1[i] - flux_c_m1[i]) / (2.0 * hc);
                    double laplacian_c  = lam * (rho_c_p1[i] - 2.0 * rho_c[i] + rho_c_m1[i]) / 2;
                    rho_c[i] += laplacian_c + divergence_c / std::sqrt(N_particles);
                    // rho_c[i] = std::max(rho_c[i], 1e-14);
                }
            }

            // QoIs
            std::vector<double> phi_vals_f(nf);
            for (int i = 0; i < nf; ++i) phi_vals_f[i] = phi_fn(xf[i]);
            for (int n = 0; n < N2; ++n) {
                double proj_f = 0.0;
                for (int i = 0; i < nf; ++i) {
                    double dev = rho_f[idx(i, n, N2)] - rho_bar_f[i];
                    proj_f += dev * phi_vals_f[i];
                }
                proj_f *= hf;
                Pf[n] = N_particles * proj_f * proj_f;
            }

            // std::vector<double> xc(nc);
            for (int i = 0; i < nc; ++i) xc[i] = i * hc;
            std::vector<double> phi_vals_c(nc);
            for (int i = 0; i < nc; ++i) phi_vals_c[i] = phi_fn(xc[i]);

            for (int n = 0; n < N2; ++n) {
                double proj_c = 0.0;
                for (int i = 0; i < nc; ++i) {
                    double dev = rho_c[idx(i, n, N2)] - rho_bar_c[i];
                    proj_c += dev * phi_vals_c[i];
                }
                proj_c *= hc;
                Pc[n] = N_particles * proj_c * proj_c;
            }
        }

        // Accumulate raw moments for MLMC driver
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
