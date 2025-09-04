#include "mlmc_test.hpp"
#include <dean_kawasaki_fe.hpp>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <functional>
#include <random>

inline int idx(int i, int n, int N2) { return i * N2 + n; }

void roll_inplace(std::vector<double>& vec, int shift, int n_points) {
    if (shift == 0) return;
    std::vector<double> tmp = vec;
    for (int i = 0; i < n_points; ++i) {
        int j = (i - shift + n_points) % n_points;
        vec[i] = tmp[j];
    }
}

void run_dean_kawasaki_fe(const int N) {
    std::cout << "Running MLMC Dean-Kawasaki - FE element noise\n\n";
    int M = 8;
    int L = 5;
    int N0 = 100;
    std::vector<double> Eps = {0.001, 0.005, 0.01, 0.05};

    std::string output_convergence_filename = "../outputs/mlmc_convergence_dk_fe.csv";
    std::string output_complexity_filename  = "../outputs/mlmc_complexity_dk_fe.csv";
    std::string output_rates_filename       = "../outputs/mlmc_rates_dk_fe.csv";

    mlmc_test(
        [=](int l, int N_samples) { return dean_kawasaki_eqn_fe_l(l, N_samples); },
        M, N, L, N0, Eps,
        output_convergence_filename,
        output_complexity_filename,
        output_rates_filename
    );
}

std::pair<std::vector<double>, std::vector<double>>
dean_kawasaki_eqn_fe_l(int l, int N) {
    std::mt19937 RNG(42 + l);
    std::normal_distribution<> Z(0.0, 1.0);

    // My problem constants
    const double Z_0 = 1.0 / 8.273782635069178;
    auto rho_0 = [&](double x) {
        double s = std::sin(x - M_PI / 2.0);
        return Z_0 * (1.0 + std::exp(-(s * s) / 2.0) / std::sqrt(2.0 * M_PI));
    };
    auto phi_fn = [&](double x) { return std::sin(x); };

    const double N_particles = 2.0e6;
    const double lam = 0.25;
    const int    batch_size = 1000;
    const double eps = 1e-14; // floor for sqrt(rho)

    // Fine grid 
    int    nf  = 1 << (l + 2);
    double hf  = 2.0 * M_PI / nf;
    double dtf = lam * hf * hf;
    int    timesteps_f = nf * nf;

    // neighbor indices
    std::vector<int> iL_f(nf), iR_f(nf);
    for (int i = 0; i < nf; ++i) {
        iL_f[i] = (i == 0 ? nf - 1 : i - 1);
        iR_f[i] = (i == nf - 1 ? 0 : i + 1);
    }

    std::vector<double> sum1(4, 0.0), sum2(2, 0.0);

    for (int N1 = 0; N1 < N; N1 += batch_size) {
        int N2 = std::min(batch_size, N - N1);

        // Fine coordinates & phi
        std::vector<double> xf(nf), phi_vals_f(nf);
        for (int i = 0; i < nf; ++i) { xf[i] = i * hf; phi_vals_f[i] = phi_fn(xf[i]); }

        // Deterministic mean (fine)
        std::vector<double> rho_bar_f(nf);
        for (int i = 0; i < nf; ++i) rho_bar_f[i] = rho_0(xf[i]);

        // Initial conditions
        std::vector<double> rho_f(nf * N2), rho_f_old(nf * N2);
        for (int i = 0; i < nf; ++i)
            for (int n = 0; n < N2; ++n)
                rho_f[idx(i, n, N2)] = rho_bar_f[i];

        // Temporaries (fine)
        std::vector<double> eta_f(nf * N2);       // assembled element load
        std::vector<double> Pf(N2, 0.0), Pc(N2, 0.0);

        if (l == 0) {
            for (int t = 0; t < timesteps_f; ++t) {
                rho_f_old = rho_f;
                std::fill(eta_f.begin(), eta_f.end(), 0.0);

                // Element noise: one normal per edge; assemble [-alpha, +alpha]
                for (int i = 0; i < nf; ++i) {
                    int ip1 = iR_f[i];
                    for (int n = 0; n < N2; ++n) {
                        double rho_edge = 0.5 * (std::max(0.0, rho_f_old[idx(i, n, N2)]) +
                                                 std::max(0.0, rho_f_old[idx(ip1, n, N2)]));
                        rho_edge = std::max(rho_edge, eps);
                        double gamma = Z(RNG);
                        double alpha = std::sqrt((dtf / N_particles) * (rho_edge / hf)) * gamma;

                        eta_f[idx(i,   n, N2)] -= alpha; // left node
                        eta_f[idx(ip1, n, N2)] += alpha; // right node
                    }
                }

                // Drift (explicit)
                for (int i = 0; i < nf; ++i) {
                    int iL = iL_f[i], iR = iR_f[i];
                    for (int n = 0; n < N2; ++n) {
                        double lap = lam * (rho_f_old[idx(iR, n, N2)]
                                         - 2.0 * rho_f_old[idx(i, n, N2)]
                                         +        rho_f_old[idx(iL, n, N2)]) * 0.5;
                        double upd = rho_f[idx(i, n, N2)] + lap + eta_f[idx(i, n, N2)] / hf;
                        rho_f[idx(i, n, N2)] = std::max(0.0, upd);
                    }
                }

                // rho_bar
                std::vector<double> rho_bar_f_p1 = rho_bar_f, rho_bar_f_m1 = rho_bar_f;
                roll_inplace(rho_bar_f_p1, -1, nf);
                roll_inplace(rho_bar_f_m1,  1, nf);
                for (int i = 0; i < nf; ++i)
                    rho_bar_f[i] += lam * (rho_bar_f_p1[i] - 2.0 * rho_bar_f[i] + rho_bar_f_m1[i]) * 0.5;
            }

            // QoI (fine)
            for (int n = 0; n < N2; ++n) {
                double inner = 0.0;
                for (int i = 0; i < nf; ++i) {
                    double dev = rho_f[idx(i, n, N2)] - rho_bar_f[i];
                    inner += dev * phi_vals_f[i];
                }
                inner *= hf;
                Pf[n] = N_particles * inner * inner;
            }
        } else {

            int    nc   = nf / 2;
            double hc   = 2.0 * M_PI / nc;
            int    timesteps_c = nc * nc;
            double dtc  = 4.0 * dtf;
            std::vector<int> iL_c(nc), iR_c(nc);
            for (int i = 0; i < nc; ++i) {
                iL_c[i] = (i == 0 ? nc - 1 : i - 1);
                iR_c[i] = (i == nc - 1 ? 0 : i + 1);
            }

            std::vector<double> xc(nc), phi_vals_c(nc);
            for (int i = 0; i < nc; ++i) { xc[i] = i * hc; phi_vals_c[i] = phi_fn(xc[i]); }

            std::vector<double> rho_bar_c(nc);
            for (int i = 0; i < nc; ++i) rho_bar_c[i] = rho_0(xc[i]);

            std::vector<double> rho_c(nc * N2), rho_c_old(nc * N2);
            for (int i = 0; i < nc; ++i)
                for (int n = 0; n < N2; ++n)
                    rho_c[idx(i, n, N2)] = rho_bar_c[i];

            std::vector<double> eta_c(nc * N2);

            for (int t = 0; t < timesteps_c; ++t) {
                // Accumulate coarse element normals from fine children over 4 fine substeps
                // I'll store sum of child gammas per coarse edge (size nc x N2)
                std::vector<double> gamma_E_accum(nc * N2, 0.0);

                for (int s = 0; s < 4; ++s) {
                    // Fine substep
                    rho_f_old = rho_f;
                    std::fill(eta_f.begin(), eta_f.end(), 0.0);

                    // Element noise on fine grid; also accumulate coarse-element gammas
                    for (int i = 0; i < nf; ++i) {
                        int ip1 = iR_f[i];
                        // Which coarse element E does this fine edge belong to?
                        // Fine edges come in pairs: e0=2E, e1=2E+1
                        int E = (i / 2) % nc; // integer division groups pairs

                        for (int n = 0; n < N2; ++n) {
                            double rho_edge = 0.5 * (std::max(0.0, rho_f_old[idx(i,   n, N2)]) +
                                                     std::max(0.0, rho_f_old[idx(ip1, n, N2)]));
                            rho_edge = std::max(rho_edge, eps);
                            double gamma = Z(RNG);
                            double alpha = std::sqrt((dtf / N_particles) * (rho_edge / hf)) * gamma;

                            // Assemble fine load
                            eta_f[idx(i,   n, N2)] -= alpha;
                            eta_f[idx(ip1, n, N2)] += alpha;

                            // Accumulate this fine gamma to its parent coarse edge
                            gamma_E_accum[idx(E, n, N2)] += gamma;
                        }
                    }

                    // Fine drift (explicit) + add FE noise
                    for (int i = 0; i < nf; ++i) {
                        int iL = iL_f[i], iR = iR_f[i];
                        for (int n = 0; n < N2; ++n) {
                            double lap = lam * (rho_f_old[idx(iR, n, N2)]
                                             - 2.0 * rho_f_old[idx(i, n, N2)]
                                             +        rho_f_old[idx(iL, n, N2)]) * 0.5;
                            double upd = rho_f[idx(i, n, N2)] + lap + eta_f[idx(i, n, N2)] / hf;
                            rho_f[idx(i, n, N2)] = std::max(0.0, upd);
                        }
                    }

                    // Mean (fine)
                    std::vector<double> rho_bar_f_p1 = rho_bar_f, rho_bar_f_m1 = rho_bar_f;
                    roll_inplace(rho_bar_f_p1, -1, nf);
                    roll_inplace(rho_bar_f_m1,  1, nf);
                    for (int i = 0; i < nf; ++i)
                        rho_bar_f[i] += lam * (rho_bar_f_p1[i] - 2.0 * rho_bar_f[i] + rho_bar_f_m1[i]) * 0.5;
                }

                // Coarse step: build coarse element normals from accumulated fine gammas
                // gamma_E_c = 0.5 * sum_over_4subs(children gammas)
                // Note: Each coarse edge had both children included in gamma_E_accum via the loop above.
                // Now I assemble coarse FE load using dtc, hc, and gamma_E_c.
                std::fill(eta_c.begin(), eta_c.end(), 0.0);
                rho_c_old = rho_c;

                for (int i = 0; i < nc; ++i) {
                    int ip1 = iR_c[i];
                    for (int n = 0; n < N2; ++n) {
                        double gamma_E_c = gamma_E_accum[idx(i, n, N2)] / std::sqrt(8.0);
                        double rho_edge_c = 0.5 * (std::max(0.0, rho_c_old[idx(i,   n, N2)]) +
                                                   std::max(0.0, rho_c_old[idx(ip1, n, N2)]));
                        rho_edge_c = std::max(rho_edge_c, eps);
                        double alpha_E = std::sqrt((dtc / N_particles) * (rho_edge_c / hc)) * gamma_E_c;

                        eta_c[idx(i,   n, N2)] -= alpha_E;
                        eta_c[idx(ip1, n, N2)] += alpha_E;
                    }
                }

                // Coarse drift + FE noise
                for (int i = 0; i < nc; ++i) {
                    int iL = iL_c[i], iR = iR_c[i];
                    for (int n = 0; n < N2; ++n) {
                        double lap = lam * (rho_c_old[idx(iR, n, N2)]
                                         - 2.0 * rho_c_old[idx(i, n, N2)]
                                         +        rho_c_old[idx(iL, n, N2)]) * 0.5;
                        double upd = rho_c[idx(i, n, N2)] + lap + eta_c[idx(i, n, N2)] / hc;
                        rho_c[idx(i, n, N2)] = std::max(0.0, upd);
                    }
                }

                // Mean (coarse)
                std::vector<double> rho_bar_c_p1 = rho_bar_c, rho_bar_c_m1 = rho_bar_c;
                roll_inplace(rho_bar_c_p1, -1, nc);
                roll_inplace(rho_bar_c_m1,  1, nc);
                for (int i = 0; i < nc; ++i)
                    rho_bar_c[i] += lam * (rho_bar_c_p1[i] - 2.0 * rho_bar_c[i] + rho_bar_c_m1[i]) * 0.5;
            }

            // QoIs
            for (int n = 0; n < N2; ++n) {
                double inner_f = 0.0;
                for (int i = 0; i < nf; ++i) {
                    double dev = rho_f[idx(i, n, N2)] - rho_bar_f[i];
                    inner_f += dev * phi_vals_f[i];
                }
                inner_f *= hf;
                Pf[n] = N_particles * inner_f * inner_f;
            }
            for (int n = 0; n < N2; ++n) {
                double inner_c = 0.0;
                for (int i = 0; i < nc; ++i) {
                    double dev = rho_c[idx(i, n, N2)] - rho_bar_c[i];
                    inner_c += dev * phi_vals_c[i];
                }
                inner_c *= hc;
                Pc[n] = N_particles * inner_c * inner_c;
            }
        }

        // MLMC moments
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

