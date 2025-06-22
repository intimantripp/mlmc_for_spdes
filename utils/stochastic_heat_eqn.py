import numpy as np

def u_scheme(T, nx, lam, n_samples=1000):
    """ 
    Uses a stochastic finite difference scheme to obtain n_samples of trajectories
    of the  stochastic heat equation.
    Current implementation assumes a uniform grid in [0, 1] with nx points and 
    homogenous boundary conditions. Could easily be modified to use a different
    grid or boundary conditions, and scale the white noise also.
    """
    dx = 1 / (nx - 1)
    dt = lam * dx**2
    nsteps = int(T / dt)
    print(f"Number of time steps: {nsteps}, dt: {dt}, dx: {dx}")
    u = np.zeros((n_samples, nx)) 
    x = np.linspace(0, 1, nx) # assumes uniform grid in [0, 1]
    i = np.arange(1, nx-1) # interior points
    std_dev = np.sqrt(dt / dx)
    for t in range(nsteps):
        if t % 1000 == 0:
            print(f"Time step {t}/{nsteps}")
        dW = np.random.randn(n_samples, nx - 2) * std_dev
        u[:, i] += lam * (u[:, i+1] - 2 * u[:, i] + u[:, i-1]) + dW
    return u
