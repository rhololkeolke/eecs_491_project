from lspiframework.simulator import Simulator as BaseSim
from lspiframework.sample import Sample
from lspiframework.policy import Policy
import numpy as np

S = 20 # number of chain states
A = 2 # number of chain actions

def chain_states():
    return S

def chain_actions():
    return A

def chain_reward():
    S = chain_states()

    # this just adds an extra reward that is never
    # used instead of translating the index
    reward = np.zeros((S, 1))

    reward[0] = 1
    reward[S-1] = 1

    return reward

class Simulator(BaseSim):
    """
    The simulator for the basic chain problem described
    in the LSTDQ paper 
    """

    def __init__(self, state=None, succprob=.9):
        """
        Sets up the simulation.

        If state is None then a random starting state is selected.
        Otherwise the specified state is used (if valid)
        """

        self.S = chain_states()
        self.A = chain_actions()

        if state is None:
            self.state = np.random.randint(self.S)

        # transition model
        self.pr = np.zeros((self.S, self.A, self.S))

        # success and failure probabilities
        self.succprob = succprob
        self.failprob = 1-succprob

        for i in range(0,self.S):
            self.pr[i, 0, max(0, i-1)] = self.succprob
            self.pr[i, 0, min(self.S-1, i+1)] = self.failprob
            self.pr[i, 1, min(self.S-1, i+1)] = self.succprob
            self.pr[i, 1, max(0, i-1)] = self.failprob

        self.rew = chain_reward()

    def execute(self, action):
        """
        Given the state of the simulator and the
        action determine the next state.

        Returns a Sample object represeting this transition
        """
        
        totprob = 0.0
        for j in [self.state-1,self.state+1]:
            newstate = max(0, min(self.S-1, j))
            totprob = totprob + self.pr[self.state,
                                        action,
                                        newstate]
            if np.random.rand() <= totprob:
                nextstate = newstate
                break

        reward = self.rew[self.state, 0]
        absorb = False

        sample = Sample(self.state, action, reward, nextstate, absorb)

        # update the simulator's state
        self.state = nextstate

        return sample
        
    def reset(self, state=None):
        """
        Resets the simulator to either the specified state
        or to a random state
        """
        
        if state is None:
            self.state = np.random.randint(S)
        else:
            self.state = state
            
    def get_actions(self):
        """
        Returns a list of possible actions
        """

        return [i for i in range(self.A)]

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
    if state < 0 or state >= S:
        raise IndexError('%i is out of bounds' % state)

    # find the starting position
    base = action * (numbasis/A)

    # compute the polynomial terms
    phi[base] = 1
    
    for i in range(1, degpol+1):
        phi[base+i] = phi[base+i-1] * (10.0*(state+1)/S)

    return phi

def initialize_policy(explore, discount, basis):
    return Policy(explore, discount, A, basis)

def uniform_samples():
    samples = []
    rew = chain_reward()
    for s in range(S):
        for a in range(A):
            for i in range(10):
                if i<9:
                    if a == 1:
                        samples.append(Sample(s, a, rew[s][0], min(S-1, s+1)))
                    else:
                        samples.append(Sample(s, a, rew[s][0], max(0, s-1)))
                elif i == 9:
                    if a == 0:
                        samples.append(Sample(s, a, rew[s][0], min(S-1, s+1)))
                    else:
                        samples.append(Sample(s, a, rew[s][0], max(0, s-1)))
                else:
                    samples.append(Sample(s, a, rew[s][0], 0, 1))

    return samples

if __name__ == '__main__':
    import lspiframework.lspi as lspi

    import pdb

    maxiter = 8
    epsilon = 10**(-5)
    samples = uniform_samples()
    discount = .9
    basis = basis_pol

    policy = initialize_policy(0, discount, basis)

    pdb.set_trace()
    
    final_policy, all_policies = lspi.lspi(maxiter,
                                           epsilon,
                                           samples,
                                           policy)