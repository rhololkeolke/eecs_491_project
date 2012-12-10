import numpy as np

class Policy(object):
    """
    Describes an agent policy
    """

    def __init__(self, explore=None, discount=None, actions=None, basis=None, policy=None):
        """
        initializes the policy
        """
        if policy is None:
            self.explore = explore
            self.discount = discount
            self.actions = actions
            self.basis = basis
            k = self.basis()
            self.weights = np.zeros((k,1))
        else:
            self.explore = policy.explore
            self.discount = policy.discount
            self.actions = policy.actions
            self.basis = policy.basis
            self.weights = np.copy(policy.weights)

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
            for i in range(1,self.actions+1):
                phi = self.basis(state, i)
                q = phi.T.dot( self.weights)[0][0]

                if q > bestq:
                    bestq = q
                    besta = [i]
                    actionphi = [phi]
                elif q == bestq:
                    besta.append(i)
                    actionphi.append(phi)

            #which = np.random.randint(len(besta))
            which = 0

            action = besta[which]
            actionphi = actionphi[which]
                
        return action, actionphi
                

