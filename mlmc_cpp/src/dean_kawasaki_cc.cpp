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
    std::string output_regression_filename = "../outputs/mlmc_regression_dk_cc.csv";

    mlmc_test(
        [=] (int l, int N_samples) { return dean_kawasaki_eqn_cc_l(l, N_samples); },
        M, N, L, N0, Eps, output_convergence_filename, output_complexity_filename,
        output_regression_filename
    );
}

std::pair<std::vector<double>, std::vector<double>>
dean_kawasaki_eqn_cc_l(int l, int N)
{
    // --- Model/observable setup ---
    const double Z_0 = 1.0 / 8.273782635069178;
    auto rho_0 = [&](double x) {
        double s = std::sin(x - M_PI / 2.0);
        return Z_0 * (1.0 + std::exp(-(s * s) / 2.0) / std::sqrt(2.0 * M_PI));
    };
    auto phi_fn = [&](double x) { return std::sin(x); };

    const double N_particles = 2.0e6;
    const double inv_sqrtN   = 1.0 / std::sqrt(N_particles);
    const double lam         = 0.25;     // dt = lam * h^2 (explicit Euler)
    const int    batch_size  = 1000;

    // --- Fine grid/time ---
    int    nf   = 1 << (l + 2);
    double hf   = 2.0 * M_PI / nf;
    double dtf  = lam * hf * hf;
    int    steps_f = nf * nf;
    const double std_f   = std::sqrt(dtf / hf);
    const double inv_2hf = 1.0 / (2.0 * hf);

    // Return containers: moments of Pf-Pc (sum1) and Pf alone (sum2)
    std::vector<double> sum1(4, 0.0);
    std::vector<double> sum2(2, 0.0);

    for (int N1 = 0; N1 < N; N1 += batch_size) {
        int N2 = std::min(batch_size, N - N1);

        // Fine grid x and phi (once per batch)
        std::vector<double> xf(nf), phi_vals_f(nf);
        for (int i = 0; i < nf; ++i) { xf[i] = i * hf; phi_vals_f[i] = phi_fn(xf[i]); }

        // Deterministic mean (initial profile only; same behavior as your code)
        std::vector<double> rho_bar_f(nf);
        for (int i = 0; i < nf; ++i) rho_bar_f[i] = rho_0(xf[i]);

        // Fine state replicated across N2 samples
        std::vector<double> rho_f(nf * N2);
        for (int i = 0; i < nf; ++i)
            for (int n = 0; n < N2; ++n)
                rho_f[i * N2 + n] = rho_bar_f[i];

        // Preallocate fine temporaries
        std::vector<double> sqrt_rho_f(nf * N2);
        std::vector<double> dWf       (nf * N2);
        std::vector<double> flux_f    (nf * N2);

        // Neighbor indices (avoid % in hot loops)
        std::vector<int> iL_f(nf), iR_f(nf);
        for (int i = 0; i < nf; ++i) {
            iL_f[i] = (i == 0 ? nf - 1 : i - 1);
            iR_f[i] = (i == nf - 1 ? 0 : i + 1);
        }

        std::vector<double> Pf(N2, 0.0), Pc(N2, 0.0);

        if (l == 0) {
            // ---------------- Base level (fine only) ----------------
            // We still use half-cell construction to keep variance scaling consistent.
            const int    num_half_cells = 2 * nf;
            const double std_half       = std::sqrt(dtf / (2.0 * hf));
            std::vector<double> half_noise(num_half_cells * N2); // reused each step

            for (int t = 0; t < steps_f; ++t) {
                // half-cell noises -> dWf
                for (int k = 0; k < num_half_cells * N2; ++k)
                    half_noise[k] = std_half * Z(RNG);

                for (int i = 0; i < nf; ++i) {
                    int h0 = (2 * i) * N2;
                    int h1 = (2 * i + 1) * N2;
                    int b  = i * N2;
                    for (int n = 0; n < N2; ++n)
                        dWf[b + n] = half_noise[h0 + n] + half_noise[h1 + n]; // Var = dtf/hf
                }

                // sqrt, flux
                for (int k = 0; k < nf * N2; ++k)
                    sqrt_rho_f[k] = std::sqrt(std::max(0.0, rho_f[k]));
                for (int k = 0; k < nf * N2; ++k)
                    flux_f[k] = sqrt_rho_f[k] * dWf[k];

                // update (no roll)
                for (int i = 0; i < nf; ++i) {
                    int iL = iL_f[i], iR = iR_f[i];
                    int base  = i * N2, baseL = iL * N2, baseR = iR * N2;
                    for (int n = 0; n < N2; ++n) {
                        int k  = base + n, kL = baseL + n, kR = baseR + n;
                        double divergence = (flux_f[kR] - flux_f[kL]) * inv_2hf;
                        double laplacian  = lam * (rho_f[kR] - 2.0 * rho_f[k] + rho_f[kL]) * 0.5;
                        rho_f[k] += laplacian + divergence * inv_sqrtN;
                    }
                }
            }

            // QoI on fine
            for (int n = 0; n < N2; ++n) {
                double proj = 0.0;
                for (int i = 0; i < nf; ++i) {
                    double dev = rho_f[i * N2 + n] - rho_bar_f[i];
                    proj += dev * phi_vals_f[i];
                }
                proj *= hf;
                Pf[n] = N_particles * proj * proj;
            }

        } else {
            // ---------------- MLMC level: half-cell coupling ----------------
            int    nc  = nf / 2;
            double hc  = 2.0 * M_PI / nc;
            int    steps_c = nc * nc;
            double dtc = 4.0 * dtf;
            (void)dtc; // (only for clarity; not used explicitly)

            const double inv_2hc = 1.0 / (2.0 * hc);

            // coarse grid & phi
            std::vector<double> xc(nc), phi_vals_c(nc);
            for (int i = 0; i < nc; ++i) { xc[i] = i * hc; phi_vals_c[i] = phi_fn(xc[i]); }

            // coarse mean and state
            std::vector<double> rho_bar_c(nc);
            for (int i = 0; i < nc; ++i) rho_bar_c[i] = rho_0(xc[i]);
            std::vector<double> rho_c(nc * N2);
            for (int i = 0; i < nc; ++i)
                for (int n = 0; n < N2; ++n)
                    rho_c[i * N2 + n] = rho_bar_c[i];

            // temporaries (coarse)
            std::vector<double> dWc       (nc * N2, 0.0);
            std::vector<double> sqrt_rho_c(nc * N2);
            std::vector<double> flux_c    (nc * N2);

            std::vector<int> iL_c(nc), iR_c(nc);
            for (int i = 0; i < nc; ++i) {
                iL_c[i] = (i == 0 ? nc - 1 : i - 1);
                iR_c[i] = (i == nc - 1 ? 0 : i + 1);
            }

            // half-cell noises: we generate fresh per fine substep
            const int    num_half_cells = 2 * nf;
            const double std_half       = std::sqrt(dtf / (2.0 * hf));
            std::vector<double> half_noise(num_half_cells * N2); // reused each substep

            for (int t = 0; t < steps_c; ++t) {
                std::fill(dWc.begin(), dWc.end(), 0.0);

                // 4 fine substeps per coarse step
                for (int s = 0; s < 4; ++s) {
                    // half-cell noises
                    for (int k = 0; k < num_half_cells * N2; ++k)
                        half_noise[k] = std_half * Z(RNG);

                    // dWf from half-cells
                    for (int i = 0; i < nf; ++i) {
                        int h0 = (2 * i) * N2;
                        int h1 = (2 * i + 1) * N2;
                        int b  = i * N2;
                        for (int n = 0; n < N2; ++n)
                            dWf[b + n] = half_noise[h0 + n] + half_noise[h1 + n];
                    }

                    // Fine update (no roll)
                    for (int k = 0; k < nf * N2; ++k)
                        sqrt_rho_f[k] = std::sqrt(std::max(0.0, rho_f[k]));
                    for (int k = 0; k < nf * N2; ++k)
                        flux_f[k] = sqrt_rho_f[k] * dWf[k];

                    for (int i = 0; i < nf; ++i) {
                        int iL = iL_f[i], iR = iR_f[i];
                        int base  = i * N2, baseL = iL * N2, baseR = iR * N2;
                        for (int n = 0; n < N2; ++n) {
                            int k  = base + n, kL = baseL + n, kR = baseR + n;
                            double divergence = (flux_f[kR] - flux_f[kL]) * inv_2hf;
                            double laplacian  = lam * (rho_f[kR] - 2.0 * rho_f[k] + rho_f[kL]) * 0.5;
                            rho_f[k] += laplacian + divergence * inv_sqrtN;
                        }
                    }

                    // Build matched coarse increment from these half-cells
                    // Each coarse cell corresponds to 4 half-cells in space this substep.
                    for (int i = 0; i < nc; ++i) {
                        int h0 = (4 * i) * N2;
                        int c0 = i * N2;
                        for (int n = 0; n < N2; ++n) {
                            // Sum 4 half-cell noises that cover coarse cell i
                            double accum = half_noise[h0 + n]
                                         + half_noise[h0 + N2 + n]
                                         + half_noise[h0 + 2 * N2 + n]
                                         + half_noise[h0 + 3 * N2 + n];
                            dWc[c0 + n] += 0.5 * accum; // scaling to match Var(dtc/hc)
                        }
                    }
                } // end 4 fine substeps

                // Coarse update with aggregated dWc
                for (int k = 0; k < nc * N2; ++k)
                    sqrt_rho_c[k] = std::sqrt(std::max(0.0, rho_c[k]));
                for (int k = 0; k < nc * N2; ++k)
                    flux_c[k] = sqrt_rho_c[k] * dWc[k];

                for (int i = 0; i < nc; ++i) {
                    int iL = iL_c[i], iR = iR_c[i];
                    int base  = i * N2, baseL = iL * N2, baseR = iR * N2;
                    for (int n = 0; n < N2; ++n) {
                        int k  = base + n, kL = baseL + n, kR = baseR + n;
                        double divergence = (flux_c[kR] - flux_c[kL]) * inv_2hc;
                        double laplacian  = lam * (rho_c[kR] - 2.0 * rho_c[k] + rho_c[kL]) * 0.5;
                        rho_c[k] += laplacian + divergence * inv_sqrtN;
                    }
                }
            } // end coarse steps

            // QoI on fine & coarse
            for (int n = 0; n < N2; ++n) {
                double proj_f = 0.0;
                for (int i = 0; i < nf; ++i) {
                    double dev = rho_f[i * N2 + n] - rho_bar_f[i];
                    proj_f += dev * phi_vals_f[i];
                }
                proj_f *= hf;
                Pf[n] = N_particles * proj_f * proj_f;
            }

            for (int n = 0; n < N2; ++n) {
                double proj_c = 0.0;
                for (int i = 0; i < nc; ++i) {
                    double dev = rho_c[i * N2 + n] - rho_bar_c[i];
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
