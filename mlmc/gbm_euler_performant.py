import numpy as np
from mlmc.mlmc_test import mlmc_test  # assumes you have a Python version of mlmc_test implemented

# GBM parameters
def gbm_euler_performant(N=100000):
    S0 = 100     
    K = 100      
    T = 1        
    r = 0.05     
    sig = 0.2    

    M = 2
    L = 10
    N0 = 1000
    Eps = [0.005, 0.01, 0.02, 0.05, 0.1]
    validation_value = 10.46

    # European Call Option
    option = 1
    mlmc_test(lambda l, N: gbm_l(l, N, option, S0, K, T, r, sig), M, N, L, N0, Eps, validate=True, validation_value=validation_value)

    # Digital Call Option
    # option = 2
    # mlmc_test(lambda l, N: gbm_l(l, N, option, S0, K, T, r, sig), M, N, L, N0, Eps, validate=False)


def gbm_l(l, N, option, S0, K, T, r, sig, batch_size=10000):
    
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

    rng = np.random.default_rng(seed=42)

    for N1 in range(0, N, batch_size):
        N2 = min(batch_size, N - N1)

        # Preallocate reusable arrays
        Sf = np.full(N2, S0, dtype=np.float64)
        Sc = Sf.copy() if l > 0 else None
        dWf = np.empty(N2)
        dWc = np.empty(N2) if l > 0 else None
        Pc = np.zeros(N2)
        Pf = np.empty(N2)

        if l == 0:
            rng.standard_normal(N2, out=dWf)
            dWf *= np.sqrt(hf)
            Sf += r * Sf * hf + sig * Sf * dWf

        else:
            dWc.fill(0.0)
            for _ in range(nc):
                for _ in range(M):
                    rng.standard_normal(N2, out=dWf)
                    dWf *= np.sqrt(hf)
                    Sf += r * Sf * hf + sig * Sf * dWf
                    dWc += dWf
                Sc += r * Sc * hc + sig * Sc * dWc
                dWc.fill(0.0)  # reset for next time step

        disc = np.exp(-r * T)
        if option == 1:
            Pf[:] = disc * np.maximum(0, Sf - K)
            if l > 0:
                Pc = disc * np.maximum(0, Sc - K)

        elif option == 2:
            Pf[:] = disc * 5.0 * (1 + np.sign(Sf - K))
            if l > 0:
                Pc = disc * 5.0 * (1 + np.sign(Sc - K))

        # Calculate moments
        diff = Pf - Pc
        diff2 = diff * diff
        diff3 = diff2 * diff
        diff4 = diff3 * diff
        sum1[0] += np.sum(diff)
        sum1[1] += np.sum(diff2)
        sum1[2] += np.sum(diff3)
        sum1[3] += np.sum(diff4)
        sum2[0] += np.sum(Pf)
        sum2[1] += np.sum(Pf**2)

    return sum1, sum2
