import numpy as np

def energy(u):
    delta_x = 1 / (u.shape[0] - 1) # Assuming u is a 2D array with shape (n, N)
    return np.sum(u**2, axis=0) * delta_x    


def nth_fourier_mode(n, u):
    # Calculates integral from 0 to 1 of 2*u(x,t)sin(n pi x) dx
    x = np.linspace(0, 1, len(u))
    sin_basis = np.sin(n * np.pi * x)[:, np.newaxis]
    integrand = 2 * u * sin_basis
    fourier_mode = np.trapz(integrand, x, axis=0)
    return fourier_mode


def analytic_fourier_mode(n):
    # Analytic solution for the nth Fourier mode of the stochastic heat equation
    return 1 / (2 * n**2 * np.pi**2) * (1 - np.exp(-2 * n**2 * np.pi**2 * 0.25))

def analytic_fourier_mode_var(n):
    true_var = (1 - np.exp(- 2 * n**2 * np.pi**2 * 0.25)) / (n**2 * np.pi**2)
    return true_var
