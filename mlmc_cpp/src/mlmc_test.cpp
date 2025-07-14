
#include "mlmc_test.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <functional>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>

void mlmc_test(
    std::function<std::pair<std::vector<double>, std::vector<double>>(int, int)> mlmc_fn,
    int M, int N, int L, int N0, const std::vector<double>& Eps,
    const std::string& output_filename
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
    std::ofstream file_out(output_filename);
    file_out << "level,ave_Pf-Pc,ave_Pf,var_Pf-Pc,var_Pf,kurtosis,check,cost\n";
    for (int i = 0; i <= L; ++i) {
        file_out << levels[i] << ","
                << del1[i] << ","
                << del2[i] << ","
                << var1[i] << ","
                << var2[i] << ","
                << kur1[i] << ","
                << chk1[i] << ","
                << cost[i] << "\n";
    }
    file_out.close();
    std::cout << "\nWrote convergence table to mlmc_convergence.csv\n";
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
