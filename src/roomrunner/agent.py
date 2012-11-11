import numpy as np
import abc

import pdb

class Agent(object):
    """
    ABC of all agents
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def update(self, prev_s, action, curr_s, reward):
        return

    @abc.abstractmethod
    def get_action(self, state, actions):
        return

    @abc.abstractmethod
    def episode_over(self):
        return

class RandomWalkAgent(Agent):
    """
    Default agent class.

    Will simply execute a random action
    """
    
    def update(self, prev_s, action, curr_s, reward):
        pass

    def get_action(self, state, actions):
        choice = np.random.random_integers(0, len(actions)-1)
        return actions[choice]

    def episode_over(self):
        pass
        
class ControlledAgent(Agent):
    """
    Listens for keyboard input and acts accordingly
    """

    def update(self, prev_s, action, curr_s, reward):
        print "Updating"
        print

    def get_action(self, state, actions):
        choice = raw_input('Enter direction ')
        while(choice not in actions):
            choice = raw_input('Enter direction ')

        for action in actions:
            if choice == action:
                return choice

    def episode_over(self):
        pass
        
class QLearningAgent(Agent):

    def __init__(self, q_values, gamma=.8, alpha=.1, decay_rate=.01):
        self.q_values = q_values
        self.gamma = gamma
        self.epsilon = 1
        self.decay_rate = decay_rate
        self.alpha = alpha

    def episode_over(self):
        pass
    
    def update(self, prev_s, action, curr_s, reward):
        maxq = self.q_values[np.where(self.q_values == np.max(self.q_values[curr_s]))]

        try:
            maxq = maxq[0]
        except TypeError:
            pass

        td_error = reward + self.gamma*maxq - self.q_values[prev_s+(action,)]
        self.q_values[prev_s+(action,)] += self.alpha*td_error
        
    def get_action(self, state, actions):
        if(np.random.rand() < self.epsilon):
            return np.random.random_integers(0, len(actions)-1)
        else:
            return np.where(self.q_values[state] == np.max(self.q_values[state]))[0][0]

    def decay_epsilon(self):
        self.epsilon *= 1-self.decay_rate
    
class QLearningRoomAgent(QLearningAgent):

    def __init__(self, h, w, num_actions):
        self.q_values = np.zeros((h,w,num_actions))
        super(QLearningRoomAgent, self).__init__(self.q_values, decay_rate=.001)

    def update(self, prev_s, action, curr_s, reward):
        prev_t = QLearningRoomAgent._totuple(prev_s)
        curr_t = QLearningRoomAgent._totuple(curr_s)

        action_num = 0
        for i,poss_action in enumerate(action[1]):
            if(poss_action == action[0]):
                action_num = i
                break

        super(QLearningRoomAgent, self).update(prev_t, action_num, curr_t, reward)

    def get_action(self, state, actions):
        state_t = QLearningRoomAgent._totuple(state)
        action = super(QLearningRoomAgent, self).get_action(state_t, actions)
        return actions[action]

    def episode_over(self):
        super(QLearningRoomAgent, self).decay_epsilon()
        
    @staticmethod
    def _totuple(a):
        try:
            return tuple(QLearningRoomAgent._totuple(i) for i in a)
        except TypeError:
            return a
    
    