import numpy as np
from mlmc.mlmc_test import mlmc_test

def run_dean_kawasaki_eqn_fe(validation_value=None):
    M = 8
    N = 1000
    L = 5
    N0 = 100
    Eps = [0.005, 0.01, 0.02, 0.05, 0.1]
    validation_value=0.5

    del1, del2, var1, var2 = mlmc_test(lambda l, N: dean_kawasaki_eqn_l(l, N), M, N, L, N0, Eps, validation_value=validation_value)


def dean_kawasaki_eqn_l(l, N):

    Z_0 = 1 / 8.273782635069178  # normalisation constant
    rho_0 = lambda x: Z_0 * (1 + np.exp(-(np.sin(x - np.pi/2)**2) / 2) / np.sqrt(2 * np.pi))
    phi_fn = lambda x: np.sin(x)
    N_particles = 2 * 10**6
    rng = np.random.default_rng(42 + l)

    lam = 0.25
    nf = 2**(l + 2)
    hf = 2 * np.pi / nf
    dtf = lam * hf**2
    timesteps_f = nf**2
    xf = np.linspace(0, 2*np.pi, nf, endpoint=False)

    if l > 0:
        nc = nf // 2
        hc = 2 * np.pi / nc
        dtc = 4 * dtf
        timesteps_c = nc**2
        xc = np.linspace(0, 2*np.pi, nc, endpoint=False)

    i_f = np.arange(nf)
    ip1_f = (i_f + 1) % nf
    if l > 0:
        i_c = np.arange(nc)
        ip1_c = (i_c + 1) % nc

    sum1 = np.zeros(4)
    sum2 = np.zeros(2)

    # Small positivity floor for multiplicative factor
    eps = 1e-14

    for N1 in range(0, N, 1000):
        N2 = min(1000, N - N1)

        # Fine initial condition
        rho_bar_f = rho_0(xf)
        rho_f = np.tile(rho_bar_f, (N2, 1)).T

        if l == 0:
            for _ in range(timesteps_f):
                rho_edge = 0.5 * (rho_f + rho_f[ip1_f, :])
                rho_edge = np.maximum(rho_edge, eps)

                # One standard normal per edge/element
                gamma_e = rng.standard_normal((nf, N2))

                # alpha_e = sqrt((dt/Np) * (rho_e/h))
                alpha_e = np.sqrt((dtf / N_particles) * (rho_edge / hf)) * gamma_e

                eta_f = np.zeros_like(rho_f)
                eta_f[i_f, :] -= alpha_e
                eta_f[ip1_f, :] += alpha_e

                # diffusive term
                laplacian = lam * (np.roll(rho_f, -1, axis=0) - 2*rho_f + np.roll(rho_f, 1, axis=0)) / 2

                # Update
                rho_f += laplacian + eta_f / hf
                rho_f = np.maximum(rho_f, 0.0)

                # rho_bar
                rho_bar_f += lam * (np.roll(rho_bar_f, -1) - 2*rho_bar_f + np.roll(rho_bar_f, 1)) / 2

            # QoI on fine
            deviation = rho_f - rho_bar_f[:, np.newaxis]
            phi_vals = phi_fn(xf)
            inner_products = hf * np.sum(deviation * phi_vals[:, np.newaxis], axis=0)
            Pf = N_particles * inner_products**2
            Pc = np.zeros(N2)

        else:
            # Coarse initial condition (independent state evolution, coupled noise)
            rho_bar_c = rho_0(xc)
            rho_c = np.tile(rho_bar_c, (N2, 1)).T  # shape (nc, N2)

            for _ in range(timesteps_c):
                # Accumulator for coarse element normals across the four fine substeps
                gamma_E_accum = np.zeros((nc, N2))

                # Four fine substeps for one coarse
                for _sub in range(4):
                    # FE element noise on fine grid 
                    rho_edge = 0.5 * (rho_f + rho_f[ip1_f, :])
                    rho_edge = np.maximum(rho_edge, eps)
                    gamma_e = rng.standard_normal((nf, N2))
                    alpha_e = np.sqrt((dtf / N_particles) * (rho_edge / hf)) * gamma_e

                    eta_f = np.zeros_like(rho_f)
                    eta_f[i_f, :] -= alpha_e
                    eta_f[ip1_f, :] += alpha_e

                    # Drift on fine
                    laplacian = lam * (np.roll(rho_f, -1, axis=0) - 2*rho_f + np.roll(rho_f, 1, axis=0)) / 2
                    rho_f += laplacian + eta_f / hf
                    rho_f = np.maximum(rho_f, 0.0)

                    # rho_bar fine
                    rho_bar_f += lam * (np.roll(rho_bar_f, -1) - 2*rho_bar_f + np.roll(rho_bar_f, 1)) / 2

                    # Build variance-preserving coarse normals from children
                    # Each coarse element E corresponds to two fine elements: e0=2E, e1=2E+1
                    e0 = 2 * i_c
                    e1 = 2 * i_c + 1
                    gamma_E_accum += gamma_e[e0, :] + gamma_e[e1, :]

                # Coarse normal per element for the coarse step
                gamma_E_c = gamma_E_accum / np.sqrt(8.0)

                rho_edge_c = 0.5 * (rho_c + rho_c[ip1_c, :])
                rho_edge_c = np.maximum(rho_edge_c, eps)

                alpha_E = np.sqrt((dtc / N_particles) * (rho_edge_c / hc)) * gamma_E_c

                eta_c = np.zeros_like(rho_c)
                eta_c[i_c, :] -= alpha_E
                eta_c[ip1_c, :] += alpha_E

                # Drift on coarse
                laplacian_c = lam * (np.roll(rho_c, -1, axis=0) - 2*rho_c + np.roll(rho_c, 1, axis=0)) / 2
                rho_c += laplacian_c + eta_c / hc
                rho_c = np.maximum(rho_c, 0.0)

                # rho_bar coarse
                rho_bar_c += lam * (np.roll(rho_bar_c, -1) - 2*rho_bar_c + np.roll(rho_bar_c, 1)) / 2

            # QoIs
            deviation = rho_f - rho_bar_f[:, np.newaxis]
            phi_vals = phi_fn(xf)
            inner_products = hf * np.sum(deviation * phi_vals[:, np.newaxis], axis=0)
            Pf = N_particles * inner_products**2

            deviation = rho_c - rho_bar_c[:, np.newaxis]
            phi_vals = phi_fn(xc)
            inner_products = hc * np.sum(deviation * phi_vals[:, np.newaxis], axis=0)
            Pc = N_particles * inner_products**2

        # MLMC accummulations
        diff = Pf - Pc
        sum1[0] += np.sum(diff)
        sum1[1] += np.sum(diff**2)
        sum1[2] += np.sum(diff**3)
        sum1[3] += np.sum(diff**4)
        sum2[0] += np.sum(Pf)
        sum2[1] += np.sum(Pf**2)

    return sum1, sum2
