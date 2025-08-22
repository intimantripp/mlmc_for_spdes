import numpy as np
from mlmc.mlmc_test import mlmc_test

# solves u_t = u_xx + xi(x,t) with u(0,t) = u(1,t) = 0. u(x,0) = 0.
# xi(x,t) is a space-time white noise, which we approximate by a sum of increments
# of a Wiener process. The increments are independent in space and time.
# Currently simulates up to T = 0.25

np.random.seed(seed=42)
def default_qoi(u):
    delta_x = 1 / (u.shape[0] - 1) # Assuming u is a 2D array with shape (n, N)
    return np.sum(u**2, axis=0) * delta_x

def default_noise_coupling(dWf):
    
    coupled_noises = (0.5 * dWf[:-2:2, :] +
                      dWf[1:-1:2, :] + 
                      0.5 * dWf[2::2, :])
    coupled_noises *= np.sqrt(1/3)
    return coupled_noises

def right_nearest_neighbour(dWf):
    num_rows = int(dWf.shape[0] / 2 - 0.5) # nf - 1 internal points for fine. Corresponds to (nf + 1) / 2 - 1 coarse internals
    dWc =  dWf[:-1, :].reshape(num_rows, 2, dWf.shape[1]).sum(axis=1) * 0.5 # sum of 2 fine adjacent increments, ignore final point
    return dWc

def nth_fourier_mode(n, u):
    # Calculates integral from 0 to 1 of 2*u(x,t)sin(n pi x) dx
    x = np.linspace(0, 1, len(u))
    sin_basis = np.sin(n * np.pi * x)[:, np.newaxis]
    integrand = 2 * u * sin_basis
    fourier_mode = np.trapz(integrand, x)
    return fourier_mode

def stoch_heat_eqn_qoi(qoi_fn=default_qoi, noise_coupling=right_nearest_neighbour, validation_value=None):
    M = 8
    N = 10000
    L = 6
    N0 = 100
    Eps = [0.005, 0.01, 0.02, 0.05, 0.1]
    if qoi_fn.__name__=='default_qoi':
        validation_value = 1/12 - np.exp(- 2 * np.pi**2) / (2 * np.pi**2)
    del1, del2, var1, var2 = mlmc_test(lambda l, N: stoch_heat_eqn_l(l, N, qoi_fn, noise_coupling), M, N, L, N0, Eps, validation_value=validation_value)
    
    return del1, del2, var1, var2


def stoch_heat_eqn_l(l, N, qoi_fn=default_qoi, noise_coupling=default_noise_coupling):
    # This is the same as the para.py module, except we apply different 
    # random increments to each spatial point in both grids.
    lam = 0.25
    nf = 2**(l + 1)
    hf = 1 / nf
    dtf = lam * hf**2
    timesteps_f = nf**2 # number of time steps for fine grid with T = 0.25
    if l > 0:
        nc = nf // 2
        hc = 1 / nc
        dtc = lam * hc**2
        timesteps_c = nc**2 # number of time steps for coarse grid with T = 0.25

    std_f = np.sqrt(dtf / hf) # calculation outside the loop
    sum1 = np.zeros(4)
    sum2 = np.zeros(2)

    for N1 in range(0, N, 100):
        rng = np.random.default_rng()   
        N2 = min(100, N - N1)
        uf = np.zeros((nf+1, N2))

        if l == 0:
            i = np.arange(1, nf) # indices of internal points, excluding the first and last
            for _ in range(timesteps_f):
                dWf = std_f * rng.standard_normal((nf - 1, N2)) # different increments for each spatial point
                uf[i, :] += lam * (uf[i + 1, :] - 2 * uf[i, :] + uf[i-1, :]) + dWf
            
            # compute the quantity of interest for the fine grid
            Pf = qoi_fn(uf)
            Pc = np.zeros(N2)
        else:
            uc = np.zeros((nc+1, N2))
            i_f = np.arange(1, nf)
            i_c = np.arange(1, nc)

            for _ in range(timesteps_c):
                dWc = np.zeros((nc-1, N2))
                for _ in range(4):
                    dWf = std_f * rng.standard_normal((nf - 1, N2))
                    uf[i_f, :] += lam * (uf[i_f+1, :] - 2 * uf[i_f, :] + uf[i_f-1, :]) + dWf
                    dWc +=  noise_coupling(dWf)
                uc[i_c, :] += lam * (uc[i_c+1, :] - 2 * uc[i_c, :] + uc[i_c-1, :]) + dWc
            Pc = qoi_fn(uc)
            Pf = qoi_fn(uf)
        
        diff = Pf - Pc
        sum1[0] += np.sum(diff)
        sum1[1] += np.sum(diff**2)
        sum1[2] += np.sum(diff**3)
        sum1[3] += np.sum(diff**4)
        sum2[0] += np.sum(Pf)
        sum2[1] += np.sum(Pf**2)
    
    return sum1, sum2 #sum1 is moments of MLMC estimator, sum2 is moments of MC estimator
