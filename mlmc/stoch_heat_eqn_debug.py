import numpy as np
from mlmc.mlmc_test import mlmc_test


def stoch_heat_eqn():
    nvert = 3
    M = 8 
    N = 1000
    L = 6
    N0 = 1000
    Eps = [0.005, 0.01, 0.02, 0.05, 0.1]

    validation_value = 0.08527441178994156
    validation_value = 4.138060043680517
    mlmc_test(lambda l, N: stoch_heat_eqn_l(l, N), M, N, L, N0, Eps, nvert,validate=True, validation_value=validation_value)
    



def stoch_heat_eqn_l(l, N):
    # This is the same as the para.py module, except we apply different 
    # random increments to each spatial point in both grids.
    lam = 0.25
    nf = 2**(l + 1)
    hf = 1 / nf
    dtf = lam * hf**2
    timesteps_f = nf**2

    if l > 0:
        nc = nf // 2
        hc = 1 / nc
        dtc = lam * hc**2
        timesteps_c = nc**2

    std_f = np.sqrt(dtf) # REMOVE
    sum1 = np.zeros(4)
    sum2 = np.zeros(2)

    for N1 in range(0, N, 100):
        N2 = min(100, N-N1)
        uf = np.zeros((nf+1, N2))

        if l == 0:
            i = np.arange(1, nf) # indices of internal points, excluding the first and last
            for _ in range(timesteps_f):
                # dWf = std_f * np.random.randn(nf - 1, N2) # different increments for each spatial point - noise
                dWf = std_f * np.random.randn(N2)
                # uf[i, :] += lam * (uf[i+1, :] - 2 * uf[i, :] + uf[i-1, :]) + dWf
                uf[i, :] += lam * (uf[i+1, :] - 2 * uf[i, :] + uf[i-1, :]) + np.tile(dWf, (nf - 1, 1))
            
            Pf = hf * np.sum(uf**2, axis=0)
            Pc = np.zeros((1, N2))
        else:
            uc = np.zeros((nc+1, N2))
            i_f = np.arange(1, nf)
            i_c = np.arange(1, nc)

            for _ in range(timesteps_c):
                # dWc = np.zeros((nc - 1, N2)) 
                dWc = np.zeros(N2)
                for _ in range(4): # 4 fine timesteps per coarse timestep
                    # dWf = std_f * np.random.randn(nf - 1, N2)
                    dWf = std_f * np.random.randn(N2)
                    # uf[i_f, :] += lam * (uf[i_f+1, :] - 2 * uf[i_f, :] + uf[i_f-1, :]) + dWf
                    uf[i_f, :] += lam * (uf[i_f+1, :] - 2 * uf[i_f, :] + uf[i_f-1, :]) + 10 * np.tile(dWf, (nf - 1, 1)) # REMOVE
                    # dWc +=  dWf[:-1, :].reshape(nc-1, 2, N2).sum(axis=1) # sum of 2 fine adjacent increments, ignore final point
                    dWc += dWf # REMOVE

                # uc[i_c, :] += lam * (uc[i_c+1, :] - 2 * uc[i_c, :] + uc[i_c-1, :]) + dWc
                uc[i_c, :] += lam * (uc[i_c+1, :] - 2 * uc[i_c, :] + uc[i_c-1, :]) + np.tile(dWc, (nc - 1, 1))
            
            Pf = hf * np.sum(uf**2, axis=0)
            Pc = hc * np.sum(uc**2, axis=0)

        diff = Pf - Pc

        sum1[0] += np.sum(diff)
        sum1[1] += np.sum(diff**2)
        sum1[2] += np.sum(diff**3)
        sum1[3] += np.sum(diff**4)
        sum2[0] += np.sum(Pf)
        sum2[1] += np.sum(Pf**2)
    
    return sum1, sum2

