
#include "mlmc_test.hpp"
#include "mlmc.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <functional>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <string>
#include <iomanip>
#include <limits>

void mlmc_test(
    std::function<std::pair<std::vector<double>, std::vector<double>>(int, int)> mlmc_fn,
    int M, int N, int L, int N0, const std::vector<double>& Eps,
    const std::string& output_convergence_filename, 
    const std::string& output_complexity_filename
) {
    std::vector<double> del1, del2, var1, var2, kur1, chk1, cost;
    std::vector<int> levels;

    // Print header with date/time
    time_t now = time(nullptr);
    char date[64];
    strftime(date, sizeof(date), "%c", localtime(&now));
    printf("\n**********************************************************\n");
    printf("*** MLMC file version 1.0     produced by              ***\n");
    printf("*** C++ mlmc_test on %s         ***\n", date);
    printf("**********************************************************\n");
    printf("\n**********************************************************\n");
    printf("*** Convergence tests, kurtosis, telescoping sum check ***\n");
    printf("*** using N =%7d samples                           ***\n", N);
    printf("**********************************************************\n");
    printf("\n l   ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)    var(Pf)    kurtosis     check        cost \n");
    printf("--------------------------------------------------------------------------------------\n");

    for (int l = 0; l <= L; ++l) {
        levels.push_back(l);
        std::cout << "l = " << l << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto [sum1, sum2] = mlmc_fn(l, N);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cost.push_back(elapsed.count());
        
        // Estimator calculation
        for (auto& x : sum1) x /= N;
        for (auto& x : sum2) x /= N;
        
        // Kurtosis calculation
        double numerator = sum1[3] - 4 * sum1[2] * sum1[0] + 6 * sum1[1] * sum1[0] * sum1[0] - 3 * std::pow(sum1[0], 4);
        double denominator = std:: pow(sum1[1] - sum1[0] * sum1[0], 2);
        double kurt = (denominator > 0) ? numerator / denominator : NAN;

        // Add elements to vectors
        del1.push_back(sum1[0]);
        del2.push_back(sum2[0]);
        var1.push_back(sum1[1] - sum1[0] * sum1[0]);
        double var = std::max(sum2[1] - sum2[0] * sum2[0], 1e-12);
        var2.push_back(var);
        kur1.push_back(kurt);

        double check = 0.0;
        if (l > 0) {
            check = std::abs(del1[l] + del2[l-1] - del2[l]) / 
            (3.0 * std::sqrt(var1[l]) + std::sqrt(var2[l-1]) + std::sqrt(var2[l]) / std::sqrt(N));
        }
        chk1.push_back(check);

        // Print formatted results for this level
        printf("%2d  %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e\n",
            l, del1[l], del2[l], var1[l], var2[l], kur1[l], chk1[l], cost[l]);
        

    }

    // Print the MLMC estimator
    double estimator = std::accumulate(del1.begin(), del1.end(), 0.0);
    std::cout << "\nMLMC estimator = " << estimator << std::endl;

    // regression estimates
    int start = std::max(2, static_cast<int>(std::floor(0.4 * levels.size())));
    std::vector<int> reg_levels(levels.begin() + start, levels.end());
    std::vector<double> reg_del1(del1.begin() + start, del1.end());
    std::vector<double> reg_var1(var1.begin() + start, var1.end());

    std::cout << "\nEstimates of key MLMC Theorem parameters based on linear regression:\n";
    double alpha = regression(reg_levels, reg_del1);
    
    double beta = regression(reg_levels, reg_var1);
    std::cout << "beta = " << beta << " (exponent for MLMC variance)\n";

    double gamma = std::log2(cost.back() / cost[cost.size() - 2]);
    std::cout << "gamma = " << gamma << " (exponent for MLMC cost)\n";

    // Consistency error and kurtosis warnings
    double max_chk = *std::max_element(chk1.begin(), chk1.end());
    if (max_chk > 1.0) {
        std::cout << "WARNING: maximum consistency error = " << max_chk << "\n";
        std::cout << "Indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied\n";
    }
    if (kur1.back() > 100.0) {
        std::cout << "WARNING: kurtosis on finest level = " << kur1.back() << "\n";
        std::cout << "Indicates MLMC correction dominated by a few rare paths; see http://mathworld.wolfram.com/SampleVarianceDistribution.html\n";
    }

    printf("\n******************************************************\n");
    printf("*** Linear regression estimates of MLMC parameters ***\n");
    printf("******************************************************\n");
    printf("\n alpha = %f  (exponent for MLMC weak convergence)\n", alpha);
    printf(" beta  = %f  (exponent for MLMC variance) \n", beta);
    printf(" gamma = %f  (exponent for MLMC cost) \n", gamma);

    // Output results to file
    std::ofstream file_out(output_convergence_filename);
    file_out << "level,ave_Pf-Pc,ave_Pf,var_Pf-Pc,var_Pf,kurtosis,check,cost,N\n";
    for (int i = 0; i <= L; ++i) {
        file_out << levels[i] << ","
                << del1[i] << ","
                << del2[i] << ","
                << var1[i] << ","
                << var2[i] << ","
                << kur1[i] << ","
                << chk1[i] << ","
                << cost[i] << ","
                << N << "\n";
    }
    file_out.close();
    std::cout << "\nWrote convergence table to " << output_convergence_filename << "\n";

    // Complexity tests: Running MLMC for different Eps values
    std::vector<std::vector<int>> Nls;
    std::vector<std::vector<double>> costs;
    std::vector<double> mlmc_estimates;
    std::vector<double> std_costs;
    std::vector<double> mlmc_costs;
    std::vector<std::vector<double>> Yls;  // Store per-level estimators for each epsilon
    std::vector<std::vector<double>> Vls; // Store per-level variances for each epsilon

    std::cout << "\n*********** Complexity test (MLMC vs standard MC for different eps) ***********\n";
    for (const auto& eps: Eps) {
        std::cout << "Running MLMC for eps = " << eps << "...\n";
        auto [P, Nl, suml] = mlmc(N0, eps, mlmc_fn, alpha, beta, gamma);

        mlmc_estimates.push_back(P);
        Nls.push_back(Nl);

        std::vector<double> means(Nl.size()), variances(Nl.size());
        for (size_t l = 0; l < Nl.size(); ++l) {
            if (Nl[l] > 0) {
                double mean = suml[0][l] / Nl[l];
                double mean_sq = suml[1][l] / Nl[l];
                means[l] = mean;
                variances[l] = std::max(0.0, mean_sq - mean * mean);
            } else {
                means[l] = 0.0;
                variances[l] = 0.0;
            }
        }
            
        Yls.push_back(means);
        Vls.push_back(variances);

        // Compute MLMC cost = sum_{l} Nl[l] * Cl[l]
        std::vector<double> Cl(Nl.size());
        for (size_t l = 0; l < Nl.size(); ++l)
            Cl[l] = std::pow(M, l);
        
        double mlmc_cost = 0.0;
        for (size_t l = 0; l < Nl.size(); ++l)
            mlmc_cost += Nl[l] * Cl[l];
        
        // Standard MC cost estimate: var2[-1] is the finest-level variance
        double var_fine = var2.back();
        double std_cost = 0.0;
        for (size_t l = 0; l < Nl.size(); ++l)
            std_cost += (2 * var_fine / (eps * eps)) * Cl[l];

        mlmc_costs.push_back(mlmc_cost);
        std_costs.push_back(std_cost);

        std::cout << " MLMC estimate = " << P << ", MLMC cost = " << mlmc_cost << ", Std MC cost = " << std_cost << std::endl;
    }

    // Output complexity data
    size_t max_levels = 0;
    for (const auto& Nl : Nls) max_levels = std::max(max_levels, Nl.size());

    std::ofstream comp_out(output_complexity_filename);
    comp_out << "eps,mlmc_estimate,mlmc_cost,std_mc_cost";
    for (size_t l = 0; l < Nls[0].size(); ++l)
        comp_out << ",Nl_" << l;
    for (size_t l = 0; l < Yls[0].size(); ++l) {
        comp_out << ",Yl_" << l;
    }
    for (size_t l = 0; l < Vls[0].size(); ++l) {
        comp_out << ",Vl_" << l;
    }
    comp_out << "\n";

    for (size_t i = 0; i < Eps.size(); ++i) {
        comp_out << Eps[i] << "," << mlmc_estimates[i] << "," << mlmc_costs[i] << "," << std_costs[i];

        // Pad with zeros if this run didn't have all levels
        for (size_t l = 0; l < max_levels; ++l)
            comp_out << "," << (l < Nls[i].size() ? Nls[i][l] : std::numeric_limits<double>::quiet_NaN());
        for (size_t l = 0; l < max_levels; ++l)
            comp_out << "," << (l < Yls[i].size() ? Yls[i][l] : std::numeric_limits<double>::quiet_NaN());
        for (size_t l = 0; l < max_levels; ++l)
            comp_out << "," << (l < Vls[i].size() ? Vls[i][l] : std::numeric_limits<double>::quiet_NaN());

        comp_out << "\n";
    }
    comp_out.close();
    std::cout << "\nWrote complexity table to " << output_complexity_filename << "\n";

}
