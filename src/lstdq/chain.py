import numpy as np
import collections

S = 20
A = 2

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

    