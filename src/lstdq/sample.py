class Sample(object):
    """
    A class for storing samples from the environment
    """

    def __init__(self, state, action, reward, nextstate, absorb=False):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextstate = nextstate
        self.absorb = absorb

    def __str__(self):
        return '(%i, %i, %i, %i, %s)' % (self.state, self.action, self.reward,
                                        self.nextstate, self.absorb)
