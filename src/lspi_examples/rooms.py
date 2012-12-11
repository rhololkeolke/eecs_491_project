"""
This module implements the functions
needed by the lspiframework and protovalueframework
for the multiple rooms problem.

The multiple rooms problem consists of an agent
in an environment composed of multiple rooms.

The agent may have a step cost associated with each move.

The moves may be probabilistic (i.e. succeeded with some
probability and moving to another state randomly with some
probability)

The agent tries to maximize its reward when placed at a random
starting location.
"""

from lspiframework.simulator import Simulator as BaseSim
from lspiframework.sample import Sample
from lspiframework.policy import Policy, RandomPolicy
import numpy as np
import yaml
from terminal import Color

    

class Simulator(BaseSim):
    """
    Simulates the rooms and the agent-environment
    interactions
    """

    __wall_sym = '*'
    __empty_sym = '-'
    __agent_sym = 'A'
    __goal_sym = 'G'

    # these values will be set by the
    # init method
    states = 0
    actions = 0
    rewards = np.zeros(1)

    def state_to_index(self, state):
        """
        Converts a state in the form of a tuple
        consisting of a row and column index in the grid
        to an index in the total number of states

        e.g.
        (0,0) becomes 0
        (h-1,w-1) becomes h*w-1
        """
        return state[1]*self.h + state[0]

    def index_to_state(self, index):
        """
        Converts an index in the total number of states
        into a state in the grid

        e.g.
        0 becomes (0,0)
        h*w-1 becomes (h-1,w-1)
        """
        col = int(index/self.h)
        row = index - col*self.h
        return (row, col)
        

    def __init__(self, path):
        """
        Takes in a configuration file path
        and creates the internal representation
        of the world
        """

        config = None
        with open(path, 'r') as f:
            config = yaml.load(f)

        if config is None:
            print "Error opening the configuration file %s" % path
            exit(1)

        # if no step cost is specified then assume 0 step cost
        self.step_cost = float(config.get('step cost', 0.0))

        # if no action set is specified then used manhattan
        if config.get('actions', 'manhattan').lower() == 'manhattan':
            Simulator.actions = 4
            self.actions = 'manhattan'
        elif config['actions'].lower() == 'diagonal':
            Simulator.actions = 8
            self.actions = 'diagonal'
        else:
            print "Invalid action set specified"
            print "Available options are manhattan and diagonal"
            exit(1)

        # set the action success probability
        self.succprob = config.get('succprob', 1.0)
        self.failprob = 1 - self.succprob

        # set the dimensions of the entire world
        # Note: outside edges are always walls
        try:
            self.h = config['height']
            self.w = config['width']
        except KeyError:
            print "You must specify a height width for the world"
            exit(1)

        # set the number of states
        Simulator.states = self.h*self.w
        # initialize the rewards vector
        Simulator.rewards = np.zeros((self.h*self.w, 1))

        # create the world grid
        self.grid = np.tile(Simulator.__empty_sym, self.h*self.w)
        self.grid = np.reshape(self.grid, (self.h, self.w))

        # set up where the walls are located in the grid
        walls = config.get('walls', None)
        if walls is not None:
            # get the horizontal walls
            horizontal = walls.get('horizontal', None)
            if horizontal is not None:
                for row in horizontal:
                    self.grid[row, :] = Simulator.__wall_sym

            # get the vertical walls
            vertical = walls.get('vertical', None)
            if vertical is not None:
                for col in vertical:
                    self.grid[:, col] = Simulator.__wall_sym

        # set up where the doors are located in the grid
        doors = config.get('doors', None)
        if doors is not None:
            for door in doors:
                self.grid[door['y'], door['x']] = Simulator.__empty_sym
            
        # set the absorbing states and their rewards
        self.absorb = {}
        absorbing = config.get('absorbing', None)
        if absorbing is not None:
            for absorb in absorbing:
                state = (absorb['y'], absorb['x'])

                if self.grid[state[0], state[1]] == Simulator.__wall_sym:
                    continue
                    
                self.absorb[state] = absorb['r']

                #self.grid[state[0], state[1]] = Simulator.__goal_sym

                index = self.state_to_index(state)
                Simulator.rewards[index] = absorb['r']


        # select a random starting state
        self.state = None
        while self.state is None:
            row = np.random.randint(self.h)
            col = np.random.randint(self.w)
            if self.grid[row, col] != Simulator.__wall_sym and \
               self.absorb.get((row, col), None) is None:
                self.state = (row, col)

    def __str__(self):
        """
        Used for displaying the grid on the command line
        """
        grid_rep = []
        grid_rep.append('%ix%i World' % (self.h, self.w))
        grid_rep.append('\n\n\n')
        for i, row in enumerate(self.grid):
            for j, elem in enumerate(row):
                if elem == Simulator.__wall_sym:
                    grid_rep.append(Color.red(' %s ' % elem))
                else:
                    if (i,j) in self.absorb:
                        if self.absorb[(i, j)] > 0:
                            grid_rep.append(Color.green(' %s ' % Simulator.__goal_sym))
                        elif self.absorb[(i, j)] < 0:
                            grid_rep.append(Color.red(' %s ' % Simulator.__goal_sym))
                        else:
                            grid_rep.append(' %s ' % Simulator.__goal_sym)
                    elif (i, j) == self.state:
                        grid_rep.append(Color.yellow(' %s ' % Simulator.__agent_sym))
                    else:
                       grid_rep.append(' %s ' % elem)
            grid_rep.append('\n')

        grid_rep.append('\n\n\n')
        
        return ''.join(grid_rep)

    def reset(self, state=None):
        """
        Resets the simulator to either the specified state
        or to a random state
        """

        if state is not None:
            if self.grid[state[0], state[1]] != Simulator.__wall_sym and \
               self.absorb.get(state, None) is None:
                self.state = state
                return

        self.state = None
        while self.state is None:
            row = np.random.randint(self.h)
            col = np.random.randint(self.w)
            if self.grid[(row, col)] != Simulator.__wall_sym and \
               self.absorb.get((row, col), None) is None:
                self.state = (row, col)

    def get_actions(self):
        """
        Returns a list of possible actions
        """
        return [a for a in range(Simulator.actions)]

    def execute(self, action):
        """
        Given the state of the simulator and the action
        determine the next state

        Returns a Sample object representing this transition
        """

        index = self.state_to_index(self.state)
        
        if np.random.rand() > self.succprob:
            # randomly pick an incorrect action
            selection = int(np.random.rand()*(Simulator.actions-1))
            if selection >= action:
                selection += 1
        else:
            selection = action

        # find the new row
        if selection in [0, 4, 7]:
            # moving up
            newrow = max(0, min(self.h-1, self.state[0]-1))
        elif selection in [2, 5, 6]:
            # moving down
            newrow = max(0, min(self.h-1, self.state[0]+1))
        else:
            # not moving up or down
            newrow = self.state[0]

        # find the new column
        if selection in [1, 4, 5]:
            # moving right
            newcol = max(0, min(self.w-1, self.state[1]+1))
        elif selection in [3, 6, 7]:
            # moving left
            newcol = max(0, min(self.w-1, self.state[1]-1))
        else:
            # not moving left or right
            newcol = self.state[1]

        if self.grid[newrow, newcol] == Simulator.__wall_sym:
            newrow = self.state[0]
            newcol = self.state[1]
            
        nextstate = (newrow, newcol)
        nextindex = self.state_to_index(nextstate)

        if nextstate in self.absorb:
            absorb = True
        else:
            absorb = False

        reward = Simulator.rewards[nextindex, 0] + self.step_cost

        sample = Sample(index, action, reward, nextindex, absorb)

        self.state = nextstate

        return sample
        
                

if __name__ == '__main__':
    import sys
    import pdb

    path = sys.argv[1]

    sim = Simulator(path)

    pdb.set_trace()

    