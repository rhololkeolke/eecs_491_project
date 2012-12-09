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
        # initialize weights to random values
        # between -1 and 1
        self.weights = np.random.rand(1,k)*2 - 1

        self.height = h
        self.width = w

        self.graph = SparseGraph(self.height*self.width)

        self.eigen_vectors = None
        self.eigen_vals = None

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

    def construct_graph(self):
        """
        Given all of the samples for this agent instance
        construct an undirected graph of states
        """
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
                if self.graph[prev_index, next_index] != 1:
                    self.graph.addEdge(prev_index, next_index)
        

    def adjacency_matrix(self):
        """
        Given all of the samples for this agent instance
        compute an adjacency matrix
        """
        return self.graph.adjacencyMatrix()
        
    def degree_matrix(self):
        """
        Given an adjacency matrix compute the degree
        matrix for it
        """
        return np.diag(self.graph.outDegreeSequence())

    def normalized_laplacian(self):
        """
        This method computes the normalized symmetric laplacian.

        L = D^(-1/2)*(D-W)*D^(-1/2)

        Where W is an adjacency matrix
        and D is the degree matrix (i.e. sum of the rows of W)
        """
        return self.graph.normalisedLaplacianSym()

    def compute_basis_functions(self):
        """
        This method computes the basis functions
        by taking the k smoothest eigen vectors of
        the combinatorial laplacian
        """

        self.eigen_vals, self.eigen_vectors = eigsh(self.normalized_laplacian(),
                                         k = self.k,
                                         #M = self.degree_matrix(),
                                         which = 'SM')
        

    def compute_A_matrix(self, i):
        """
        This method computes the A matrix approximation
        using the stored eigenfunctions and samples

        See the paper for the formula
        """
        A = np.zeros((1,self.k))

        for sample in trajectories[i]:
            

    def compute_b_matrix(self, i):
        """
        This method computes the b matrix approximation
        using the stored eigenfunctions and samples

        See the paper for the formula
        """
        pass

    def get_new_weights(self, i):
        """
        This method will compute the new weights
        based off of the A matrix
        """
        pass

    @staticmethod
    def _totuple(a):
        try:
            return tuple(QLearningRoomAgent._totuple(i) for i in a)
        except TypeError:
            return a