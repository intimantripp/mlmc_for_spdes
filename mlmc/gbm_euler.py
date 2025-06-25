import numpy as np
from mlmc.mlmc_test import mlmc_test  # assumes you have a Python version of mlmc_test implemented

# GBM parameters
def gbm_euler():
    S0 = 100     
    K = 100      
    T = 1        
    r = 0.05     
    sig = 0.2    

    nvert = 3
    M = 2
    N = 100000
    L = 8
    N0 = 1000
    Eps = [0.005, 0.01, 0.02, 0.05, 0.1]

    # European Call Option
    option = 1
    mlmc_test(lambda l, N: gbm_l(l, N, option, S0, K, T, r, sig), M, N, L, N0, Eps, nvert)

    # Digital Call Option
    option = 2
    mlmc_test(lambda l, N: gbm_l(l, N, option, S0, K, T, r, sig), M, N, L, N0, Eps, nvert)


def gbm_l(l, N, option, S0, K, T, r, sig):
    
    M = 2
    nf = M**l
    if l == 0:
        nc = 1  # dummy value, won’t be used
        hc = T  # dummy value, won’t be used
    else:
        nc = M**l // M
        hc = T / nc

    hf = T / nf

    sum1 = np.zeros(4)
    sum2 = np.zeros(2)

    for N1 in range(0, N, 10000):
        N2 = min(10000, N - N1)
        Sf = np.full(N2, S0, dtype=float)
        Sc = Sf.copy()
        Pc = np.zeros(N2)

        if l == 0:
            dWf = np.sqrt(hf) * np.random.randn(N2)
            Sf += r * Sf * hf + sig * Sf * dWf

        else:
            for _ in range(nc):
                dWc = np.zeros(N2)
                for _ in range(M):
                    dWf = np.sqrt(hf) * np.random.randn(N2)
                    dWc += dWf
                    Sf += r * Sf * hf + sig * Sf * dWf
                Sc += r * Sc * hc + sig * Sc * dWc

        if option == 1:
            Pf = np.exp(-r * T) * np.maximum(0, Sf - K)
            if l > 0:
                Pc = np.exp(-r * T) * np.maximum(0, Sc - K)

        elif option == 2:
            Pf = np.exp(-r * T) * 5 * (1 + np.sign(Sf - K))
            if l > 0:
                Pc = np.exp(-r * T) * 5 * (1 + np.sign(Sc - K))


        sum1[0] += np.sum(Pf - Pc)
        sum1[1] += np.sum((Pf - Pc)**2)
        sum1[2] += np.sum((Pf - Pc)**3)
        sum1[3] += np.sum((Pf - Pc)**4)
        sum2[0] += np.sum(Pf)
        sum2[1] += np.sum(Pf**2)

    return sum1, sum2
