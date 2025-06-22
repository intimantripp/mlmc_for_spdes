import numpy as np
from mlmc_test import mlmc_test


def para():
    nvert = 3
    M = 8 # gamma = 3, 2**(3*l) cost per level
    N = 1000
    L = 6
    N0 = 100
    Eps = [0.005, 0.01, 0.02, 0.05, 0.1]
    mlmc_test(lambda l, N: para_l(l, N), M, N, L, N0, Eps, nvert)

    


def para_l(l, N):
    lam = 0.25 #timestep/spacing^2 - CFL number, for parabolic PDEs critical value is 1/2
    nf = 2**(l + 1) # number of spatial subintervals in the fine grid, 2^l + 1 points
    hf = 1 / nf # spatial step size
    dtf = lam * hf**2 # timestep size
    timesteps_f = nf**2

    if l > 0:
        nc = nf // 2
        hc = 1 / nc
        dtc = lam * hc**2
        timesteps_c = nc**2
    
    sum1 = np.zeros(4)
    sum2 = np.zeros(2)

    for N1 in range(0, N, 100): # batch sizes of 100 used up to N samples
        N2 = min(100, N-N1) # ensures we choose the right number of samples, actual batch size
        uf = np.zeros((nf+1, N2)) # nf + 1 because nf is number of subintervals

        if l == 0:
            i = np.arange(1, nf) # indices of internal points
            for _ in range(timesteps_f): #for each
                dWf = np.sqrt(dtf) * np.random.randn(N2)
                uf[i, :] += lam * (uf[i+1, :] - 2 * uf[i, :] + uf[i-1, :]) \
                      + 10 * np.tile(dWf, (nf - 1, 1))
            Pf = hf * np.sum(uf**2, axis=0)
            Pc = np.zeros((1, N2))
        else:
            uc = np.zeros((nc+1, N2))
            i_f = np.arange(1, nf)
            i_c = np.arange(1, nc)

            for _ in range(timesteps_c):
                dWc = np.zeros(N2)
                for _ in range(4): # 4 fine timesteps per coarse timestep
                    dWf = np.sqrt(dtf) * np.random.randn(N2)
                    dWc += dWf # sum of 4 brownian increments has same variance as one coarse increment
                    uf[i_f, :] += lam * (uf[i_f+1, :] - 2 * uf[i_f, :] + uf[i_f-1, :]) \
                          + 10 * np.tile(dWf, (nf - 1, 1))
                uc[i_c, :] += lam * (uc[i_c+1, :] - 2 * uc[i_c, :] + uc[i_c-1, :]) \
                      + 10 * np.tile(dWc, (nc - 1, 1))
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
