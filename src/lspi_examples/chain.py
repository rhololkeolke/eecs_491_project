from lspiframework.simulator import Simulator as BaseSim
from lspiframework.sample import Sample
from lspiframework.policy import Policy, RandomPolicy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mpl

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

    def get_transition_model(self):
        """
        Returns the transition model
        """
        return self.pr

    def get_reward_model(self):
        """
        Returns the reward model
        """
        return self.rew

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

def collect_samples(maxepisodes=10, maxsteps=500, policy=None):
    """
    Collects samples from the simulator using the policy
    by running it at most maxepisodes episodes of each which
    is at most maxsteps steps long
    """

    if policy is None:
        policy = RandomPolicy(1.0, 0.0, A)

    sim = Simulator()

    samples = []
    
    for episode in range(maxepisodes):
        # reset the simulator
        sim.reset()

        for step in range(maxsteps):
            samples.append(sim.execute(policy.select_action(sim.state)[0]))
            # if this was an absorbing state
            # then start a new episode
            if samples[-1].absorb:
                break

    return samples

def solve(policy):
    """
    Given a policy this function evaluates it
    """
    sim = Simulator()
    pr = sim.get_transition_model()
    rew = sim.get_reward_model()

    dim = len(rew)

    pa0 = np.squeeze( pr[:, 0, :])
    pa1 = np.squeeze( pr[:, 1, :])

    prob = np.zeros((dim, dim))
    for i in range(1, dim+1):
        a = policy.select_action(i)[0]
        if a == 0:
            prob[i,:] = pa0[i, :]
        else:
            prob[i,:] = pa1[i, :]

    v = np.inv(np.eye(dim) - policy.discount*prob).dot(rew)

    q1 = rew + policy.discount * pa0.dot(v)
    q2 = rew + policy.discount * pa1.dot(v)
    q = max(q1, q2)

    return v, q1, q2, q

def display_policy(policy, figure=None):
    """
    Plots the policy as a color map over all the states.

    If a plot handle is provided it will use that plot.
    Otherwise it will create a new plot

    returns the plot handle that it used
    """
    S = chain_states()
    actions = np.zeros((1, S))

    for i in range(S):
        actions[0,i] = policy.select_action(i)[0]

    figure = plt.figure()
    plt.imshow(actions)
    cmap = plt.colorbar(boundaries=[0,1,2], values=[0,1])
    plt.title('Policy Actions')
    plt.xlabel('State')
    plt.show()
    
    
    
if __name__ == '__main__':
    import lspiframework.lspi as lspi
    import protovalueframework.pvf as pvf

    import pdb

    k = 10
    maxiter = 20
    epsilon = 10**(-5)
    #samples = uniform_samples()
    samples = collect_samples()
    discount = .9

    # construct a graph from the samples
    graph = pvf.construct_graph(samples, S)

    basis = pvf.create_basis_function(graph, S, A, k)
    
    policy = initialize_policy(0, discount, basis)

    final_policy, all_policies = lspi.lspi(maxiter,
                                           epsilon,
                                           samples,
                                           policy)

    display_policy(final_policy)
    pdb.set_trace()