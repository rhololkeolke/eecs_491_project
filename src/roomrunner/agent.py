import numpy as np

class Agent(object):
    """
    Default agent class.

    Will simply execute a random action
    """
    
    def update(self, prev_s, action, curr_s):
        pass

    def get_action(self, state, actions):
        choice = np.random.random_integers(0, len(actions)-1)
        return actions[choice]

class QLearningAgent(object):
    import numpy as np
    pass

    
    