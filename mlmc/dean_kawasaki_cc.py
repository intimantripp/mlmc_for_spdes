import numpy as np
from mlmc.mlmc_test import mlmc_test

def dean_kawasaki_eqn_cc(validation_value=None):
    M = 8
    N = 1000
    L = 4
    N0 = 100
    Eps = [0.01, 0.02, 0.05, 0.1]

    del1, del2, var1, var2 = mlmc_test(lambda l, N: dean_kawasaki_eqn_cc_l(l, N), M, N, L, N0, Eps, validation_value=validation_value)



def dean_kawasaki_eqn_cc_l(l, N):

    Z_0 = 1 / 8.273782635069178 # normalisation constant
    rho_0 = lambda x: Z_0 * (1 + np.exp(-(np.sin(x - np.pi/2)**2) / 2) / np.sqrt(2 * np.pi))
    phi_fn = lambda x: np.sin(x)
    N_particles = 2 * 10**6

    lam = 0.25
    nf = 2**(l+2)
    hf = 2*np.pi / nf
    dtf = lam * hf**2
    timesteps_f = nf**2
    xf = np.linspace(0, 2*np.pi, nf, endpoint=False)

    if l > 0:
        nc = nf // 2
        hc = 2*np.pi / nc
        timesteps_c = nc**2
        xc = np.linspace(0, 2*np.pi, nc, endpoint=False)
    
    std_f = np.sqrt(dtf / hf)
    sum1 = np.zeros(4)
    sum2 = np.zeros(2)

    for N1 in range(0, N, 1000):
        N2 = min(1000, N - N1)
        rho_bar_f = rho_0(xf)
        rho_f = np.tile(rho_bar_f, (N2, 1)).T

        if l == 0:
            for _ in range(timesteps_f):
                sqrt_rho = np.sqrt(np.maximum(rho_f, 0))
                dW = std_f * rng.standard_normal((nf, N2))
                flux = sqrt_rho * dW
                divergence = (np.roll(flux, -1, axis=0) - np.roll(flux, 1, axis=0)) / (2*hf)

                laplacian = lam * (np.roll(rho_f, -1, axis=0) - 2*rho_f + np.roll(rho_f, 1, axis=0)) / 2
                rho_f += laplacian + divergence / np.sqrt(N_particles)
                rho_bar_f += lam * (np.roll(rho_bar_f, -1, axis=0) - 2*rho_bar_f + np.roll(rho_bar_f, 1, axis=0)) / 2

            deviation = rho_f - rho_bar_f[:, np.newaxis]
            phi_vals = phi_fn(xf)
            inner_products = hf * np.sum(deviation * phi_vals[:, np.newaxis], axis=0)
            Pf = N_particles * inner_products**2
            Pc = np.zeros(N2)
        else:
            rho_bar_c = rho_0(xc)
            rho_c = np.tile(rho_bar_c, (N2, 1)).T
            for _ in range(timesteps_c):
                dWc = np.zeros((nc, N2))
                for _ in range(4):
                    num_half_cells = 2 * nf
                    std_half = np.sqrt(hf * dtf / 2)
                    half_cell_noises = rng.standard_normal((num_half_cells, N2)) * std_half
                    dWf = (half_cell_noises[::2, :] + half_cell_noises[1::2, :]) / hf

                    sqrt_rho = np.sqrt(np.maximum(rho_f, 0))
                    flux = sqrt_rho * dWf
                    divergence = (np.roll(flux, -1, axis=0) - np.roll(flux, 1, axis=0)) / (2 * hf)

                    laplacian = lam * (np.roll(rho_f, -1, axis=0) - 2*rho_f + np.roll(rho_f, 1, axis=0)) / 2
                    rho_f += laplacian + divergence / (np.sqrt(N_particles))
                    rho_bar_f += lam * (np.roll(rho_bar_f, -1, axis=0) - 2*rho_bar_f +  np.roll(rho_bar_f, 1, axis=0)) / 2

                    dWc += (half_cell_noises[0::4, :] + half_cell_noises[1::4, :] +
                                              half_cell_noises[2::4, :] + half_cell_noises[3::4, :]) / hc
                sqrt_rho = np.sqrt(np.maximum(rho_c, 0))
                flux = sqrt_rho * dWc
                divergence = (np.roll(flux, -1, axis=0) - np.roll(flux, 1, axis=0)) / (2 * hc)
                laplacian = lam * (np.roll(rho_c, -1, axis=0) - 2*rho_c + np.roll(rho_c, 1, axis=0)) / 2
                rho_c += laplacian + divergence / np.sqrt(N_particles)
                rho_bar_c += lam * (np.roll(rho_bar_c, -1, axis=0) - 2 * rho_bar_c + np.roll(rho_bar_c, 1, axis=0)) / 2

            deviation = rho_f - rho_bar_f[:, np.newaxis]
            phi_vals = phi_fn(xf)
            inner_products = hf * np.sum(deviation * phi_vals[:, np.newaxis], axis=0)
            Pf = N_particles * inner_products**2

            deviation = rho_c - rho_bar_c[:, np.newaxis]
            phi_vals = phi_fn(xc)
            inner_products = hc * np.sum(deviation * phi_vals[:, np.newaxis], axis=0)
            Pc = N_particles * inner_products**2

        diff = Pf - Pc
        sum1[0] += np.sum(diff)
        sum1[1] += np.sum(diff**2)
        sum1[2] += np.sum(diff**3)
        sum1[3] += np.sum(diff**4)
        sum2[0] += np.sum(Pf)
        sum2[1] += np.sum(Pf**2)

    return sum1, sum2
