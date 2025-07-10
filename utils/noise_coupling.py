def default_noise_coupling(dWf):
    # Approach adopted in the Dean-Kwasaki paper #
    coupled_noises = (0.5 * dWf[:-2:2, :] +
                      dWf[1:-1:2, :] + 
                      0.5 * dWf[2::2, :])
    coupled_noises *= np.sqrt(1/3)
    return coupled_noises

def suspect_noise_coupling(dWf):
    # Pretty useless. Violates independent increments for coarse grid. #
    num_rows = int(dWf.shape[0] / 2 - 0.5) # nf - 1 internal points for fine. Corresponds to (nf + 1) / 2 - 1 coarse internals
    dWc =  dWf[:-1, :].reshape(num_rows, 2, dWf.shape[1]).sum(axis=1) * 0.5 # sum of 2 fine adjacent increments, ignore final point
    return dWc
