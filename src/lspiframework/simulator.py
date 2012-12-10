"""
This module provides the basic abstract base class for a simulator
"""

import abc

class Simulator(object):
    """
    ABC of all simulators
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self, state=None):
        """
        Resets the simulator to the specificed state.
        If the specified state is None then a random
        state is chosen
        """
        return

    @abc.abstractmethod
    def execute(self, action):
        """
        Executes the specified action

        Returns a sample object
        """

    @abc.abstractmethod
    def get_actions(self):
        """
        Returns a list of possible actions
        """
        return