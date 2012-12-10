import numpy as np
from policy import Policy
from sample import Sample
import lspi
import pdb

S = 20
A = 2
rew = np.zeros((S+1,1))
rew[1] = 1
rew[S] = 1

def basis_pol(state=None, action=None):
    """
    Computes a set of polynomial (on "state") basis functions
    up to a certain degree. The set is duplicated for each action.
    The action determines which segment will be active
    """

    degpol = 4; # degree of the polynomial

    numbasis = (degpol+1) * A

    if state is None or action is None:
        return numbasis

    # initialize
    phi = np.zeros((numbasis, 1))

    # check if stat is within bounds
    if state < 1 or state > S:
        raise IndexError('%i is out of bounds' % state)

    # find the starting position
    base = (action-1) * (numbasis/A)

    # compute the polynomial terms
    phi[base] = 1

    for i  in range(1, degpol+1):
        phi[base+i] = phi[base+i-1] * (10.0*state/S)


    return phi

def create_chain_policy(explore, discount, basis):
    return Policy(explore, discount, A, basis)

def uniform_samples():
    samples = []
    for s in range(1,S+1):
        for a in range(1,A+1):
            for i in range(0,10):
                if i< 9:
                    if a == 2:
                        samples.append(Sample(s, a, rew[s], min(S, s+1)))
                    else:
                        samples.append(Sample(s, a, rew[s], max(1,s-1)))
                elif i == 9:
                    if a == 1:
                        samples.append(Sample(s,a,rew[s],min(S,s+1)))
                    else:
                        samples.append(Sample(s,a,rew[s],max(1,s-1)))
                else:
                    samples.append(Sample(s,a,rew[s],0,1))
    return samples
        

if __name__ == '__main__':
    import lspi

    maxiter = 8
    epsilon = 10**(-5)
    samples = uniform_samples()
    discount = .9
    basis = basis_pol

    
    policy = create_chain_policy(0, discount, basis)

    pdb.set_trace()

    final_policy, all_policies = lspi.lspi(maxiter, epsilon, samples, basis, discount, policy)