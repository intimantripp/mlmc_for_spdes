import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec  as gridspec
from mlmc.mlmc import mlmc

        

def mlmc_test(mlmc_fn, M, N, L, N0, Eps, nvert, validate=False, validation_value=None, **mlmc_l_kwargs):
    """
    Runs a MLMC test for a given function mlmc_fn, number of levels L,
    number of samples N, initial sample size N0, and a list of desired accuracies Eps.
    The function mlmc_fn should take two arguments: level l and number of samples N.
    It should return two arrays: sum1 and sum2, where sum1 contains the first four moments
    of the fine level and sum2 contains the first two moments of the coarse level.
    The function will plot the results and return final mlmc estimate.
    """
    del1 = []; del2 = [] #del1 = P_l - P_{l-1}, del2 = P_l
    var1 = []; var2 = [] #var1 = var(P_l - P_{l-1}), var2 = var(P_l)
    kur1 = []
    chk1 = []
    cost = [] # cost of each level
    L = np.arange(0, L+1)

    for l in L:
        print(f"l = {l}")
        start_time = time.time()
        sum1, sum2 = mlmc_fn(l, N, **mlmc_l_kwargs)
        time_taken = time.time() - start_time
        cost.append(time_taken)
        sum1 = sum1 / N
        sum2 = sum2 / N
        # kurtosis
        numerator = sum1[3]- 4 * sum1[2] * sum1[0]+ 6 * sum1[1] * sum1[0]**2- 3 * sum1[0]**4
        denominator = (sum1[1] - sum1[0]**2)**2
        kurt = numerator / denominator if denominator > 0 else np.nan

        del1.append(sum1[0])
        del2.append(sum2[0])
        var1.append(sum1[1] - sum1[0]**2)
        var = max(sum2[1] - sum2[0]**2, 1e-12) #handles var=0 cases
        var2.append(var)
        kur1.append(kurt)

        if l==0:
            check = 0
        else:
            check = abs(del1[l] + del2[l-1] - del2[l]) /  \
                (3.0 * (np.sqrt(var1[l])) + np.sqrt(var2[l-1]) + np.sqrt(var2[l])/np.sqrt(N))
        chk1.append(check)

    # convert items to numpy arrays so I can do index selecting
    del1 = np.array(del1)
    del2 = np.array(del2)
    var1 = np.array(var1)
    var2 = np.array(var2)
    kur1 = np.array(kur1)
    chk1 = np.array(chk1)

    # use linear regression to estimate alpha, beta and gamma
    start = max(2, int(np.floor(0.4 * len(L))))
    range_ = np.arange(start, len(L))

    print('Estimates of key MLMC Theorem parameters based on linear regression: ')

    pa = np.polyfit(L[range_], np.log2(abs(del1[range_])), 1)
    alpha = -pa[0]
    print(f"alpha = {alpha:.6f} (exponent for MLMC weak convergence)")

    pb = np.polyfit(L[range_], np.log2(abs(var1[range_])), 1)
    beta = -pb[0]
    print(f"beta = {beta:.6f} (exponent for MLMC variance)")

    gamma = np.log2(cost[-1]/cost[-2])
    print(f"gamma = {gamma:.6f} (exponent for MLMC cost")

    # Perform checks on consistency error and kurtosis
    if max(chk1) > 1:
        print(f"WARNING: maximum consistency error = {max(chk1):.6f}")
        print(f"Indicates identitiy E[Pf-Pc] = E[Pf] - E[Pc] not satisfied")

    if kur1[-1] > 100:
        print(f"WARNING: kurtosis on finest level = {kur1[-1]:.6f}. \n \
                indicates MLMC correction dominated by a few rare paths; for information \
                on the connection to variance of sample variances see \
                http://mathworld.wolfram.com/SampleVarianceDistribution.html")

    # Plot figures
    nrows = nvert + 2 if validate else nvert + 1
    nrows = 5
    fig = plt.figure(figsize=(10, 3.5 * nrows))
    gs = gridspec.GridSpec(nrows, 2, figure=fig)
    axs = []
    for row in range(nrows - 2): # rows 0, 1, 2
        axs.append(fig.add_subplot(gs[row, 0]))
        axs.append(fig.add_subplot(gs[row, 1]))

    var_ax, mean_ax, chk_ax, kur_ax, count_ax, cost_ax = axs
    # MLMC and MC validation plot
    validation_ax = fig.add_subplot(gs[3, :])
    
    # Results plot
    results_ax = fig.add_subplot(gs[4, :])

    var_ax.plot(L, np.log2(var2), '-*', label='P_l')
    var_ax.plot(L[1:], np.log2(var1[1:]), '--*', label='P_l - P_{l-1}')
    var_ax.set_xlabel(r'level $l$')
    var_ax.set_ylabel(r'$\log_2$ variance')
    var_ax.legend(loc='upper right')

    mean_ax.plot(L, np.log2(np.abs(del2)), '-*', label='P_l')
    mean_ax.plot(L[1:], np.log2(np.abs(del1[1:])), '--*', label='P_l - P_{l-1}')
    mean_ax.set_xlabel(r'level $l$')
    mean_ax.set_ylabel(r'$\log_2 |\text{mean}|$')
    mean_ax.legend(loc='upper right')

    if nvert == 3:
        chk_ax.plot(L[1:] - 1e-9, chk1[1:], '--*')
        chk_ax.set_xlabel(r'level $l$')
        chk_ax.set_ylabel('consistency check')

        kur_ax.plot(L[1:] - 1e-9, kur1[1:], '--*')
        kur_ax.set_xlabel(r'level $l$')
        kur_ax.set_ylabel('kurtosis')

        # Plot target value of estimator
        if validation_value is not None:
            validation_ax.axhline(y=validation_value, linestyle='--', color='crimson', linewidth=2, label='Target QoI')

        # MLMC estimates
        mlmc_estimator = np.cumsum(del1)
        mlmc_se = np.sqrt(np.cumsum(var1) / N)
        validation_ax.errorbar(L, mlmc_estimator, yerr=mlmc_se, fmt='o-', capsize=5, elinewidth=1.2, color='blue', ecolor='lightblue',
                               label=r'MLMC estimates $\pm \sigma$ ')
        
        # Plot E[P_l] for context
        mc_se = np.sqrt(var2 / N)
        validation_ax.errorbar(L, del2, yerr=mc_se, fmt='s--', elinewidth=1.2, color='gray', ecolor='silver', label=r'MC estimates $\pm \sigma$')

        #Formatting
        validation_ax.set_xlabel(r'Level $\ell$', fontsize=12)
        validation_ax.set_ylabel(r'Estimate', fontsize=12)
        validation_ax.tick_params(axis='both', which='major', labelsize=10)
        validation_ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        validation_ax.spines['top'].set_visible(False)
        validation_ax.spines['right'].set_visible(False)
        validation_ax.set_title('Convergence of MLMC Estimate', fontsize=13)
        validation_ax.legend(loc='best', frameon=True, fontsize=11)


    if nvert == 1:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))
    
    # Run MLMC for different EPS
    Nls = []
    ls = []
    maxl = 0
    mlmc_cost = []
    std_cost = []
    mlmc_solns = []

    for i, eps in enumerate(Eps):
        print(f"eps = {eps}")
        gamma = np.log2(M)

        # Run MLMC
        P, Nl = mlmc(N0, eps, mlmc_fn, alpha, beta, gamma, **mlmc_l_kwargs)
        l = len(Nl) - 1
        maxl = max(maxl, l) # See how many levels were needed to get our desired level of eps

        # Compute cost estimates - compares mlmc cost with standard mc cost
        levels = np.arange(0, l + 1)
        mlmc_c = (1 + 1 / M) * np.sum(Nl * M**levels) #C_tot = sum of Cl*Nl
        std_c = np.sum((2 * var2[-1] / eps**2) * M**levels) #var2 contains powers of P^f

        mlmc_cost.append(mlmc_c)
        std_cost.append(std_c)

        # Store per-level sample counts
        Nls.append(Nl)
        ls.append(levels)

        # Store mlmc solutions
        mlmc_solns.append(P)

    # Now make sure all the arrays we will be plotting are the same size
    for j in range(len(Eps)):
        current_Nl = Nls[j]
        current_ls = ls[j]

        pad_len = maxl + 1 - len(current_Nl)

        if pad_len > 0:
            Nls[j] = np.pad(current_Nl, (0, pad_len), constant_values=current_Nl[-1])
            ls[j] = np.pad(current_ls, (0, pad_len), constant_values=current_ls[-1])
    
    
    Nls_array = np.column_stack(Nls)  # shape: (maxl+1, len(Eps))
    ls_array = np.column_stack(ls)

    count_ax.set_prop_cycle(None)
    for i in range(len(Eps)):
        count_ax.semilogy(ls_array[:, i], Nls_array[:, i], '--o', label=f'$\\varepsilon$ = {Eps[i]:.3g}')
    count_ax.set_xlabel('level $\\ell$')
    count_ax.set_ylabel('$N_\\ell$')
    count_ax.legend()

    eps_squared = np.array(Eps) ** 2
    cost_ax.set_prop_cycle(None)
    cost_ax.loglog(Eps, eps_squared * std_cost, '--o', label='Std MC')
    cost_ax.loglog(Eps, eps_squared * mlmc_cost, '--x', label='MLMC')
    cost_ax.set_xlabel('accuracy $\\epsilon$')
    cost_ax.set_ylabel('$\\epsilon^2$ Cost')
    cost_ax.legend(loc='upper right')

    # arrange Eps in ascending order to show error getting smaller and smaller
    results_ax.plot(Eps, mlmc_solns, '--*', label='MLMC estimates')
    results_ax.fill_between(
        Eps,
        np.array(mlmc_solns) - np.array(Eps),
        np.array(mlmc_solns) + np.array(Eps),
        color='crimson',
        alpha=0.2,
        label=r'$\pm \epsilon^2$'
    )
    results_ax.set_xlabel(r'standard error, $\epsilon^2$')
    results_ax.set_ylabel(r'MLMC estimate')
    results_ax.invert_xaxis()
    if validation_value is not None:
        results_ax.axhline(y=validation_value, linestyle='--', color='crimson', linewidth=2, 
                           label='True QoI')
    results_ax.legend()

    plt.tight_layout()
    plt.show()

    return del1, del2, var1, var2
