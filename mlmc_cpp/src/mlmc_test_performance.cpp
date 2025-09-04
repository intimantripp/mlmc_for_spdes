// mlmc_test_performance.cpp
#include "mlmc_test_performance.hpp"
#include "mlmc.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static inline std::string eps_to_token(double eps) {
    std::ostringstream oss; oss << std::setprecision(16) << eps;
    std::string s = oss.str();
    for (auto& c : s) { if (c=='.') c='p'; else if (c=='-') c='m'; else if (c=='+') c='p'; }
    return s;
}

void mlmc_test_performance(
    std::function<std::pair<std::vector<double>, std::vector<double>>(int, int)> mlmc_fn,
    int M,
    int N0,
    const std::vector<double>& Eps,
    double alpha, double beta, double gamma,
    const std::string& timings_dir,
    const std::string& aggregate_csv,
    int loops_per_eps
) {
    using clock_t = std::chrono::high_resolution_clock;

    if (loops_per_eps < 1) loops_per_eps = 1;

    std::cout << "\n================ MLMC PERFORMANCE RUN ================\n"
              << "Using analytic coefficients: alpha=" << alpha
              << ", beta=" << beta << ", gamma=" << gamma << "\n"
              << "N0=" << N0 << ", M=" << M << ", loops_per_eps=" << loops_per_eps << "\n"
              << "Writing per-eps timings to: " << (timings_dir.empty() ? "(disabled)" : timings_dir) << "\n";

    // For optional aggregate CSV
    std::vector<double> agg_eps, agg_P, agg_mlmc_cost, agg_std_cost, agg_wall_s;
    std::vector<std::vector<int>>    agg_Nls;
    std::vector<std::vector<double>> agg_Yls, agg_Vls;

    for (double eps : Eps) {
        const std::string fname = timings_dir.empty()
            ? std::string()
            : (timings_dir + "/timing_eps_" + eps_to_token(eps) + ".csv");

        std::ofstream per_eps_out;
        if (!timings_dir.empty()) {
            per_eps_out.open(fname, std::ios::out | std::ios::trunc);
            per_eps_out << "run_index,eps,wall_time_s,alpha,beta,gamma,N0,mlmc_estimate,mlmc_cost,std_mc_cost,L_used";
            // Columns sized per run, so we’ll write Nl/Yl/Vl with the run’s L_used each row
            per_eps_out << ",Nl_list,Yl_list,Vl_list\n";
        }

        for (int r = 0; r < loops_per_eps; ++r) {
            std::cout << "\n[eps=" << eps << " | run " << (r+1) << "/" << loops_per_eps << "] Running MLMC...\n";

            auto t0 = clock_t::now();
            auto [P, Nl, suml] = mlmc(N0, eps, mlmc_fn, alpha, beta, gamma);
            auto t1 = clock_t::now();
            const double wall_s = std::chrono::duration<double>(t1 - t0).count();

            // Per-level stats
            const size_t L_used = Nl.size();
            std::vector<double> means(L_used, 0.0), variances(L_used, 0.0);
            for (size_t l = 0; l < L_used; ++l) {
                if (Nl[l] > 0) {
                    const double m1 = suml[0][l] / static_cast<double>(Nl[l]);
                    const double m2 = suml[1][l] / static_cast<double>(Nl[l]);
                    means[l] = m1;
                    variances[l] = std::max(0.0, m2 - m1*m1);
                }
            }

            // Cost proxies
            double mlmc_cost = 0.0;
            for (size_t l = 0; l < L_used; ++l)
                mlmc_cost += static_cast<double>(Nl[l]) * std::pow(static_cast<double>(M), static_cast<int>(l));

            const double var_fine = (L_used > 0 ? variances.back() : 0.0);
            double std_cost = 0.0;
            for (size_t l = 0; l < L_used; ++l)
                std_cost += (2.0 * var_fine / (eps * eps)) * std::pow(static_cast<double>(M), static_cast<int>(l));

            std::cout << " Done in " << wall_s << " s. P=" << P
                      << ", MLMC_cost~" << mlmc_cost << ", StdMC_cost~" << std_cost << "\n";

            // Write a row to per-eps CSV
            if (per_eps_out.is_open()) {
                per_eps_out << r << ","
                            << std::setprecision(16) << eps << ","
                            << std::setprecision(16) << wall_s << ","
                            << alpha << "," << beta << "," << gamma << ","
                            << N0 << ","
                            << std::setprecision(16) << P << ","
                            << std::setprecision(16) << mlmc_cost << ","
                            << std::setprecision(16) << std_cost << ","
                            << (L_used == 0 ? -1 : static_cast<int>(L_used) - 1);

                // Write lists compactly in one cell each (semicolon-separated)
                auto dump_list = [](std::ostream& os, const auto& vec) {
                    os << "\"";
                    for (size_t i = 0; i < vec.size(); ++i) {
                        if (i) os << ';';
                        os << vec[i];
                    }
                    os << "\"";
                };

                per_eps_out << ",";
                dump_list(per_eps_out, Nl);
                per_eps_out << ",";
                dump_list(per_eps_out, means);
                per_eps_out << ",";
                dump_list(per_eps_out, variances);
                per_eps_out << "\n";
            }

            // Optional aggregate (one entry per *run*, not per ε)
            if (!aggregate_csv.empty()) {
                agg_eps.push_back(eps);
                agg_P.push_back(P);
                agg_mlmc_cost.push_back(mlmc_cost);
                agg_std_cost.push_back(std_cost);
                agg_wall_s.push_back(wall_s);
                agg_Nls.push_back(Nl);
                agg_Yls.push_back(means);
                agg_Vls.push_back(variances);
            }
        }

        if (per_eps_out.is_open()) {
            per_eps_out.close();
            std::cout << " Wrote " << fname << "\n";
        }
    }

    // Optional aggregate CSV (one row per run across all ε)
    if (!aggregate_csv.empty()) {
        // Find widest level count to pad columns
        size_t maxL = 0;
        for (const auto& v : agg_Nls) maxL = std::max(maxL, v.size());

        std::ofstream out(aggregate_csv, std::ios::out | std::ios::trunc);
        out << "eps,mlmc_estimate,mlmc_cost,std_mc_cost,mlmc_wall_time";
        for (size_t l = 0; l < maxL; ++l) out << ",Nl_" << l;
        for (size_t l = 0; l < maxL; ++l) out << ",Yl_" << l;
        for (size_t l = 0; l < maxL; ++l) out << ",Vl_" << l;
        out << "\n";

        out << std::setprecision(16);
        for (size_t i = 0; i < agg_eps.size(); ++i) {
            out << agg_eps[i] << ","
                << agg_P[i] << ","
                << agg_mlmc_cost[i] << ","
                << agg_std_cost[i] << ","
                << agg_wall_s[i];

            for (size_t l = 0; l < maxL; ++l)
                out << "," << (l < agg_Nls[i].size() ? agg_Nls[i][l] : std::numeric_limits<double>::quiet_NaN());
            for (size_t l = 0; l < maxL; ++l)
                out << "," << (l < agg_Yls[i].size() ? agg_Yls[i][l] : std::numeric_limits<double>::quiet_NaN());
            for (size_t l = 0; l < maxL; ++l)
                out << "," << (l < agg_Vls[i].size() ? agg_Vls[i][l] : std::numeric_limits<double>::quiet_NaN());
            out << "\n";
        }
        out.close();
        std::cout << "\nWrote aggregate performance table to " << aggregate_csv << "\n";
    }

    std::cout << "\n================ PERFORMANCE RUN COMPLETE =============\n";
}
