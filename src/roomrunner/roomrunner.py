import numpy as np
import matplotlib.pyplot as plt
import yaml

from terminal import Color
import os

from agent import Agent

import pdb

class RoomRunner(object):
    __wall_sym = '*'
    __empty_sym = '-'
    __agent_sym = 'A'

    actions = ['up', 'right', 'down', 'left']
    
    def __init__(self, path):
        config = None
        with open(path, 'r') as f:
            config = yaml.load(f)
            if(config['type'] != 'simple'):
                print "RoomRunner requires a configuration of type 'simple'. " + \
                      "Received a type of %s" % config['type']
                sys.exit(1)
            self.goals = config['goals']

        # create the grid representing
        # this simple 2 room world
        self.__create_grid(config['height'], config['width'])

        # add the walls
        self.__add_walls(config['walls'])

        # add the doors
        self.__add_doors(config['doors'])

        # create the goals
        self.__create_goals(config['goals'])

    def __create_grid(self, height, width):
        # create an empty grid
        self.grid = np.tile(RoomRunner.__empty_sym, width*height)
        self.grid = np.reshape(self.grid, (height, width))

        # create walls along the edges
        self.grid[:, 0] = RoomRunner.__wall_sym
        self.grid[:, width-1] = RoomRunner.__wall_sym
        self.grid[0, :] = RoomRunner.__wall_sym
        self.grid[height-1, :] = RoomRunner.__wall_sym

    def __add_walls(self, walls):
        for wall in walls:
            if('x' in wall):
                self.grid[:, wall['x']-1] = RoomRunner.__wall_sym
            else:
                self.grid[wall['y']-1, :] = RoomRunner.__wall_sym

    def __add_doors(self, doors):
        for door in doors:
            x = door['x']
            y = door['y']
            self.grid[y-1, x-1] = RoomRunner.__empty_sym
                

    def __create_goals(self, goals):
        self.goals = {}
        for goal in goals:
            self.goals[(goal['y']-1, goal['x']-1)] = goal['r']

    def __str__(self):
        grid_rep = ''
        for i,row in enumerate(self.grid):
            for j,elem in enumerate(row):
                if(elem == RoomRunner.__wall_sym):
                    grid_rep += Color.red(' %s ' % elem)
                elif(elem == RoomRunner.__agent_sym):
                    grid_rep += Color.yellow(' %s ' % elem)
                else:
                    if((i,j) in self.goals):
                        grid_rep += Color.green(' %s ' % elem)
                    else:
                        grid_rep += ' %s ' % elem
            grid_rep += '\n'
        return grid_rep


    def run_policy(self, agent, num_eps=1, visualize=True):
        total_rewards = 0
        total_steps = 0

        print
        print "Running %i episodes" % num_eps
        print
        for eps in range(1, num_eps+1):
            init_x = np.random.random_integers(0, self.grid.shape[0]-1)
            init_y = np.random.random_integers(0, self.grid.shape[1]-1)

            while(self.grid[init_y, init_x] == RoomRunner.__wall_sym):
                init_x = np.random.random_integers(0, self.grid.shape[0]-1)
                init_y = np.random.random_integers(0, self.grid.shape[1]-1)
            
            curr_state = np.array([init_x, init_y])
            
            self.grid[init_y, init_x] = RoomRunner.__agent_sym

            while((curr_state[0], curr_state[1]) not in self.goals):
                total_steps += 1
                selected_action = agent.get_action(curr_state, RoomRunner.actions)
                
                curr_state = self.__execute_action(curr_state, selected_action)

                if(visualize):
                    os.system('cls' if os.name=='nt' else 'clear')
                    print "episode %i" % eps
                    print
                    print self

                if((curr_state[0], curr_state[1]) in self.goals):
                    reward = self.goals[(curr_state[0], curr_state[1])]
                    print
                    print "Goal state (%i, %i) reached. Reward of %i obtained" % (
                           curr_state[0], curr_state[1], reward)
                          
                    total_rewards += reward
                    break

            self.grid[curr_state[1], curr_state[0]] = RoomRunner.__empty_sym
                
        return (total_rewards, total_steps)

    def learn_policy(self, agent, num_eps=1, visualize=True):
        print
        print "Learning from %i episodes" % num_eps
        print
        for eps in range(1, num_eps+1):
            if(visualize):
                print "episode %i" % eps
                print
                print self

    def __execute_action(self, state, action):
        if action == 'up':
            pass
        elif action == 'right':
            pass
        elif action == 'down':
            pass
        elif action == 'left':
            pass
        else:
            pass
        return state.copy()
            

if __name__ == "__main__":
    import sys, curses

    if len(sys.argv) != 2:
        print "A filename is required"
        sys.exit(1)

    path = sys.argv[1]

    

    rr = RoomRunner(path)

    a = Agent()
    
    rr.run_policy(a, num_eps=1)
    