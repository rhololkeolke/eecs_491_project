import numpy as np
import abc

import pdb

from apgl.graph import SparseGraph
from scipy.sparse.linalg import eigsh

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

    def __init__(self, q_values, gamma=.8, alpha=.1, decay_rate=.01, glie=True):
        self.q_values = q_values
        self.gamma = gamma
        self.epsilon = 1
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.glie=glie

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
        if(self.glie and np.random.rand() < self.epsilon):
            return np.random.random_integers(0, len(actions)-1)
        else:
            return np.where(self.q_values[state] == np.max(self.q_values[state]))[0][0]

    def decay_epsilon(self):
        self.epsilon *= 1-self.decay_rate
    
class QLearningRoomAgent(QLearningAgent):

    def __init__(self, h, w, num_actions, glie=True):
        self.q_values = np.zeros((h,w,num_actions))
        super(QLearningRoomAgent, self).__init__(self.q_values, decay_rate=.001, glie=glie)

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
    

class ProtoValueRoomAgent(Agent):
    """
    This class implements the basic idea of proto-value
    reinforcement learning
    """
    def __init__(self, h, w, num_actions, collect_samples=True, k=5):
        self.collect_samples = collect_samples
        # trajectories is a list of lists
        # the outer lists stores all trajectories
        # each inner list is a single trajectory
        # each sample in a trajectory is a tuple
        # (s, a, s', r)
        self.trajectories = [[]]

        # number of eigenvectors to use as
        # basis functions
        self.k = k

        # weight matrix
        self.weights = np.zeros((1,k))

        self.height = h
        self.width = w

        self.eigen_vectors = np.zeros((h*w, k))

    def episode_over(self):
        self.trajectories.append([])

    def update(self, prev_s, action, curr_s, reward):
        prev_t = ProtoValueRoomAgent._totuple(prev_s)
        curr_t = ProtoValueRoomAgent._totuple(curr_s)

        action_num = 0
        for i, poss_action in enumerate(action[1]):
            if(poss_action == action[0]):
                action_num = i
                break

        self.trajectories[-1].append((prev_t, action_num, curr_t, reward))
            
    def get_action(self, state, actions):
        # if the agent is in the sample
        # collection phase then simply choose
        # a random action
        if(self.collect_samples):
            choice = np.random.random_integers(0, len(actions)-1)
            return actions[choice]


    def compute_adjacency(self):
        """
        Given all of the samples for this agent instance
        compute an adjacency matrix
        """

        adjacency = np.zeros((self.height*self.width, self.height*self.width))
        # loop through all trajectories in samples
        # for each state transition make the cell
        # 1 if for that state
        #
        # the states are linearized so that the adjacency
        # matrix is 2D
        for trajectory in self.trajectories:
            for sample in trajectory:
                prev_state = sample[0]
                next_state = sample[2]
                prev_index = prev_state[1]*self.height + prev_state[0]
                next_index = next_state[1]*self.height + next_state[0]
                
                # not sure if its okay to have 1's in the diaganols
                adjacency[prev_index, next_index] = 1
                adjacency[next_index, prev_index] = 1

        return adjacency

    def compute_degree_matrix(self, adjacency):
        """
        Given an adjacency matrix compute the degree
        matrix for it
        """
        degree = np.diag(sum(adjacency.T))
        return degree

    def compute_laplacian(self):
        """
        This method computes the combinatorial laplacian.

        L = (D-W)

        Where W is an adjacency matrix
        and D is the degree matrix (i.e. sum of the rows of W)
        """
        adjacency = self.compute_adjacency()
        degree_matrix = self.compute_degree_matrix(adjacency)

        return degree_matrix - adjacency

    def compute_basis_functions(self):
        """
        This method computes the basis functions
        by taking the k smoothest eigen vectors of
        the combinatorial laplacian
        """
        pass
        
        

    @staticmethod
    def _totuple(a):
        try:
            return tuple(QLearningRoomAgent._totuple(i) for i in a)
        except TypeError:
            return a