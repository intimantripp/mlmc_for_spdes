import numpy as np

def mlmc(N0, eps, mlmc_l, alpha_0, beta_0, gamma, **mlmc_l_kwargs):

    alpha = max(0, alpha_0)
    beta = max(0, beta_0)

    L = 2
    Nl = np.zeros(L+1, dtype=int)
    dNl = np.full(L+1, N0, dtype=int)
    suml = np.zeros((2, L+1)) # rows: sum(Y), sum(Y^2)

    while np.sum(dNl) > 0: #while we need increase the number of samples
        for l in range(L+1):
            if dNl[l] > 0:
                sum1, sum2 = mlmc_l(l, dNl[l], **mlmc_l_kwargs)
                Nl[l] += dNl[l]
                suml[0, l] += sum1[0]
                suml[1, l] += sum1[1]
    
        # compute absolute means and variances
        ml = np.abs(suml[0,:])/Nl #get abs mean of each Y
        Vl = np.maximum(0, suml[1,:]/Nl - ml**2) # Get maximum variance from levels so far

        # fix to cope with possible zero values for ml and Vl - if ml or Vl were zero, we update them using decay estimates
        for l in range(3, L + 2):
            ml[l-1] = max(ml[l-1], 0.5 * ml[l-2] / 2**alpha)
            Vl[l-1] = max(Vl[l-1], 0.5 * Vl[l-2] / 2**beta)
        

        # use linear regression to estimate alpha and beta if they weren't updated
        L_vector = np.arange(1, L+1).reshape(-1, 1)
        if alpha_0 <= 0:
            A = np.hstack([L_vector, np.ones_like(L_vector)])
            x = np.linalg.lstsq(A, np.log2(ml[1:]), rcond=None)[0]
            alpha = max(0.5, -x[0]) # why do we take max(0.5, empirical alpha)? Answer at bottom of page 24 in notes
            print(f"Estimated alpha: {alpha:.6f}")

        if beta_0 <= 0:
            A = np.hstack([L_vector, np.ones_like(L_vector)])
            x = np.linalg.lstsq(A, np.log2(Vl[1:]), rcond=None)[0]
            beta = max(0.5, -x[0])
            print(f"Estimated beta: {beta:.6f}")

        
        # Why not calculate gamma again here Mike?

        # Calculate optimal number of additional samples
        Cl = 2**(gamma*np.arange(0, L+1))
        Ns = np.ceil(2 * np.sqrt(Vl/Cl) * np.sum(np.sqrt(Vl*Cl)) / eps**2).astype(int)
        dNl = np.maximum(0, Ns - Nl)

        # if (almost) converged, estimate remaining error and decide whether a new level is required
        # This is the remaining error estimate on page 21 being used
        if np.sum(dNl > 0.01 * Nl) == 0: #if the additional number of samples is less than 1% of current samples for any level
            idx_range = np.array([-2, -1, 0])
            rem = np.max(ml[L+idx_range]*2**(alpha*idx_range)) / (2**alpha - 1) # formula for error less than epsilon/sqrt(2)

            if rem > eps / np.sqrt(2):
                print(f"Adding new level {L+1} with remaining error {rem:.6f} > eps/sqrt(2) = {eps/np.sqrt(2):.6f}")
                L += 1
                Vl = np.append(Vl, (Vl[L-1]) / 2**beta) #Append estimate for variance at next level (var decay)
                Nl = np.append(Nl, 0)
                suml = np.column_stack([suml, [0, 0]])
                
                L_vector = np.arange(0, L+1)
                Cl = 2**(gamma * L_vector)
                term = np.sqrt(Vl / Cl)
                normaliser = np.sum(np.sqrt(Vl * Cl))
                Ns = np.ceil(2 * term * normaliser / eps**2).astype(int)
                dNl = np.maximum(0, Ns - Nl)
                print(f"New level {L} added, with {dNl[L]} samples and variance {Vl[L]:.6f}")


    P = np.sum(suml[0, :] / Nl)

    return P, Nl

