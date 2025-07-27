#include <vector>
#include <tuple>
#include <cmath>
#include <numeric>
#include <iostream>
#include <functional>
#include <algorithm>
#include "mlmc.hpp"
#include "gbm_euler.hpp"


using LevelFunction = std::function<std::pair<std::vector<double>, std::vector<double>>(int, int)>;

// Helper: elementwise sqrt for std::vector
std::vector<double> sqrt_vec(const std::vector<double>& v) {
    std::vector<double> out(v.size());
    std::transform(v.begin(), v.end(), out.begin(), [](double x){ return std::sqrt(x); });
    return out;
}

// Helper: elementwise product for std::vector
std::vector<double> mul_vec(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i]*b[i];
    return out;
}

// Helper: elementwise division for std::vector
std::vector<double> div_vec(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i]/b[i];
    return out;
}

// Helper: elementwise max for two vectors
std::vector<double> max_vec(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = std::max(a[i], b[i]);
    return out;
}

std::tuple<double, std::vector<int>, std::vector<std::vector<double>>> mlmc(
    int N0,
    double eps,
    LevelFunction mlmc_l,
    double alpha_0,
    double beta_0,
    double gamma
) {
    double alpha = std::max(0.0, alpha_0);
    double beta = std::max(0.0, beta_0);

    int L = 2;
    std::vector<int> Nl(L + 1, 0);
    std::vector<int> dNl(L + 1, N0);
    std::vector<std::vector<double>> suml(2, std::vector<double>(L + 1, 0.0)); //rows: sum(Y), sum(Y^2)

    while (std::accumulate(dNl.begin(), dNl.end(), 0) > 0) {
        for (int l = 0; l <= L; ++l) {
            if (dNl[l] > 0) {
                auto [sum1, sum2] = mlmc_l(l, dNl[l]);
                Nl[l] += dNl[l]; 
                suml[0][l] += sum1[0];
                suml[1][l] += sum1[1];
            }
        }

        // Compute means and variances
        std::vector<double> ml(L + 1, 0.0), Vl(L + 1, 0.0);
        for (int l = 0; l <= L; ++l) {
            ml[l] = std::abs(suml[0][l]) / std::max(1, Nl[l]);
            double mean = suml[0][l] / std::max(1, Nl[l]);
            double mean_sq = suml[1][l] / std::max(1, Nl[l]);
            Vl[l] = std::max(0.0, mean_sq - mean * mean);
        }

        // Fix for possible zero values
        for (int l = 3; l <= L; ++l) {
            if (l - 1 < (int)ml.size())
                ml[l-1] = std::max(ml[l-1], 0.5 * ml[l-2] / std::pow(2.0, alpha));
            if (l - 1< (int)Vl.size())
                Vl[l-1] = std::max(Vl[l-1], 0.5 * Vl[l-2] / std::pow(2.0, beta));
        }

        // Linear regressions if needed
        if (alpha_0 <= 0) {
            std::vector<int> x; std::vector<double> y;
            for (int l = 1; l <= L; ++l) {
                x.push_back(l);
                y.push_back(ml[l]);
            }
            alpha = std::max(0.5, regression(x, y));
            std::cout << "Estimated alpha: " << alpha << "\n";
        }

        if (beta_0 <= 0) {
            std::vector<int> levels_for_reg;
            std::vector<double> Vl_for_reg;
            for (int l = 1; l <= L; ++l) {
                levels_for_reg.push_back(l);
                Vl_for_reg.push_back(Vl[l]);
            }
            double slope = regression(levels_for_reg, Vl_for_reg);
            beta = std::max(0.5, slope);
            std::cout << "Estimated beta: " << beta << "\n";
        }

        // Calculate optimal number of additional samples
        std::vector<double> Cl(L + 1, 0.0);
        double sum_sqrt_VlCl = 0.0;
        for (int l = 0; l <= L; ++l) {
            Cl[l] = std::pow(2.0, gamma * l);
            sum_sqrt_VlCl += std::sqrt(Vl[l] * Cl[l]);
        }

        std::vector<int> Ns(L + 1, 0), new_dNl(L + 1, 0);
        for (int l = 0; l <= L; ++l) {
            double term = std::sqrt(Vl[l] / Cl[l]);
            Ns[l] = static_cast<int>(std::ceil(2 * term * sum_sqrt_VlCl / (eps * eps)));
            new_dNl[l] = std::max(0, Ns[l] - Nl[l]);
        }

        dNl = new_dNl;

        // Check remaining samples count and whether to add new level
        int large_update_count = 0;
        for (int l = 0; l <= L; ++l) {
            if (Nl[l] > 0 && dNl[l] > 0.01 * Nl[l])
                ++large_update_count;
        }

        if (large_update_count == 0) {
            std::vector<int> idx_range = {L-2, L-1, L};
            double rem = 0.0;
            for (int i = 0; i < (int)idx_range.size(); ++i) {
                int idx = idx_range[i];
                if (idx >= 0 && idx <= L) {
                    rem = std::max(rem, ml[idx] * std::pow(2.0, alpha * (idx - L))); // formula for error less than epsilon/2
                }
            }
            rem /= (std::pow(2.0, alpha) -1);

            if (rem > eps / std::sqrt(2.0)) {
                std::cout << "Adding new level " << L+1 << " with remaining error " << rem
                        << " > eps / sqrt(2)  = " << (eps/std::sqrt(2.0)) << std::endl;
                L += 1;
                Vl.push_back(Vl[L-1] / std::pow(2.0, beta));
                Nl.push_back(0);
                for (auto&row : suml) row.push_back(0.0);

                // Recalculate Cl, term, normalised, Ns and dNl
                std::vector<double> Cl(L + 1, 0.0);
                double sum_sqrt_VlCl = 0.0;
                for (int l = 0; l <= L; ++l) {
                    Cl[l] = std::pow(2.0, gamma * l);
                    sum_sqrt_VlCl += std::sqrt(Vl[l] * Cl[l]);
                }
                std::vector<int> Ns(L + 1, 0);
                std::vector<int> new_dNl(L + 1, 0);
                for (int l = 0; l <= L; ++l) {
                    double term = std::sqrt(Vl[l] / Cl[l]);
                    Ns[l] = static_cast<int>(std::ceil(2 * term * sum_sqrt_VlCl / (eps * eps)));
                    new_dNl[l] = std::max(0, Ns[l] - Nl[l]);
                }
                dNl = new_dNl;
            }
        }
    }
    double P = 0.0;
    for (int l = 0; l <= L; ++l) {
        if (Nl[l] > 0)
        P += suml[0][l] / Nl[l];
    }

    return std::make_tuple(P, Nl, suml);
}

double regression(const std::vector<int>& x, const std::vector<double>& y) {
    int n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = 0; i < n; ++i) {
        double log2y = std::log2(std::abs(y[i]));
        sum_x += x[i];
        sum_y += log2y;
        sum_xx += x[i] * x[i];
        sum_xy += x[i] * log2y;
    }
    double denom = n * sum_xx - sum_x * sum_x;
    double slope = (n * sum_xy - sum_x * sum_y) / denom;
    // Alpha is the negative slope
    return -slope;
}

