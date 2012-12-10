"""
This module contains the sample class.
"""

class Sample(object):
    """
    Represents a sample from the problem being learned
    """

    def __init__(self, state, action, reward, nextstate, absorb=False):
        """
        Initialize a new sample.

        By default assume that this is a non absorbing state
        """

        self.state = state
        self.action = action
        self.reward = reward
        self.nextstate = nextstate
        self.absorb = absorb

    def __repr__(self):
        return "(%s, %s, %s, %s, %s)" % (self.state,
                                         self.action,
                                         self.reward,
                                         self.nextstate,
                                         self.absorb)