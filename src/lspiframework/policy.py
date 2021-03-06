"""
This module provides the basic abstract base class for a policy
"""
import numpy as np

class Policy(object):
    """
    Represents an agent's policy
    """

    def __init__(self, explore, discount, actions, basis):
        """
        Initializes a policy
        """
        self.explore = float(explore)
        self.discount = float(discount)
        self.actions = actions
        self.basis = basis
        k = self.basis()
        self.weights = np.zeros((k, 1))

    @classmethod
    def copy(cls, policy):
        new_policy = cls(policy.explore, policy.discount,
                         policy.actions, policy.basis)
        new_policy.weights = np.copy(policy.weights)
        return new_policy

    def select_action(self, state):
        """
        Computes this policy at the given state

        Returns the action that is picked and the evaluation
        of the basis at the pair (state, action)
        """

        # should this policy explore or not?
        if np.random.rand() < self.explore:
            action = np.random.randint(self.actions)
            actionphi = self.basis(state, action)
        else:
            # pick maximum Q value action (argmax a)
            bestq = float('-inf')
            besta = []
            actionphi = []

            # find the actions with maximum Q-value
            for i in range(self.actions):
                phi = self.basis(state, i)
                q = phi.T.dot(self.weights)[0][0]

                if q > bestq:
                    bestq = q
                    besta = [i]
                    actionphi = [phi]
                elif q == bestq:
                    besta.append(i)
                    actionphi.append(phi)

            which = np.random.randint(len(besta))
            #which = 0

            action = besta[which]
            actionphi = actionphi[which]
                
        return action, actionphi

    def __repr__(self):
        output = []
        output.append('explore = %f' % self.explore)
        output.append('discount = %f' % self.discount)
        output.append('actions = %i' % self.actions)
        output.append('basis = %s' % self.basis)
        output.append('weights = %s' % self.weights)

        return '\n'.join(output)


class RandomPolicy(Policy):
    """
    Uses a basis function of 1 so that random actions are performed
    """

    @staticmethod
    def basis(state=None, action=None):
        """
        Just a dummy basis function
        that will allow this policy to act randomly
        """
        return np.ones(1)

    @classmethod
    def copy(cls, policy):
        new_policy = cls(policy.explore,
                         policy.discount,
                         policy.actions)
        new_policy.weights = np.copy(policy.weights)
        return new_policy
    
    def __init__(self, explore, discount, actions):
        super(RandomPolicy, self).__init__(explore, discount,
                                           actions, fake_basis)

class ValuePolicy(object):
    """
    Uses a value function to select the actions performed
    """

    def __init__(self, sim, T, R, V, discount):
        """
        Initializes a new value policy
        """
        self.T = T
        self.R = R
        self.V = V
        self.sim = sim
        self.discount = discount

    def select_action(self, index):
        """
        Computes this policy at the given state

        Returns the action that is picked and the evaluation
        of the basis at the pair (state, action)
        """
        # select whichever action
        state = self.sim.index_to_state(index)

        bestv = float('-inf')
        besta = []

        for a in range(self.sim.actions):
            sprimes = [index, self.sim.get_next_state(state, a)[1]]
            sprimes = list(set(sprimes))

            Va = 0
            for sprime in sprimes:
                Va += self.T[index, sprime, a]*(self.R[index, sprime] +
                                                self.discount*self.V[sprime, 0])

            if Va > bestv:
                bestv = Va
                besta = [a]
            elif Va == bestv:
                besta.append(a)

        which = np.random.randint(len(besta))
        return besta[which], bestv

def fake_basis(state=None, action=None):
    """
    Just a dummy basis function
    that will allow this policy to act randomly
    """
    return np.ones(1)
