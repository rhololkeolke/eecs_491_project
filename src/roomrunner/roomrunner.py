import numpy as np
import matplotlib.pyplot as plt
import yaml

import pickle

from terminal import Color
import os

from agent import Agent, ControlledAgent, QLearningRoomAgent

from time import sleep

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

        self.step_cost = config.get('step_cost',0)

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


    def run_episodes(self, agent, num_eps=1, learn=True, visualize=True):
        total_rewards = 0
        total_steps = 1

        print
        print "Running %i episodes" % num_eps
        print
        for eps in range(1, num_eps+1):
            init_x = np.random.random_integers(0, self.grid.shape[0]-1)
            init_y = np.random.random_integers(0, self.grid.shape[1]-1)

            while(self.grid[init_y, init_x] == RoomRunner.__wall_sym or \
                  (init_y, init_y) in self.goals):
                init_x = np.random.random_integers(0, self.grid.shape[0]-1)
                init_y = np.random.random_integers(0, self.grid.shape[1]-1)
            
            curr_state = np.array([init_x, init_y])
            
            self.grid[init_y, init_x] = RoomRunner.__agent_sym

            if(visualize):
                os.system('cls' if os.name=='nt' else 'clear')
                print "episode %i" % eps
                print
                print self
                print
                print "current state: %s" % curr_state
                print
                sleep(.5)            

            while((curr_state[0], curr_state[1]) not in self.goals):
                total_steps += 1
                selected_action = agent.get_action(curr_state, RoomRunner.actions)

                self.grid[curr_state[1], curr_state[0]] = RoomRunner.__empty_sym

                prev_state = curr_state
                
                curr_state = self.__execute_action(curr_state, selected_action)


                
                self.grid[curr_state[1], curr_state[0]] = RoomRunner.__agent_sym
                

                if(visualize):
                    os.system('cls' if os.name=='nt' else 'clear')
                    print "episode %i" % eps
                    print
                    print "executing action %s" % selected_action
                    print
                    print self
                    print
                    print "previous state: %s" % prev_state
                    print
                    print "current state: %s" % curr_state
                    print
                    sleep(.1)
                    #raw_input("Press Enter to continue...")
                else:
                    print "episode %i" % eps

                if((curr_state[0], curr_state[1]) in self.goals):
                    reward = self.goals[(curr_state[0], curr_state[1])]
                    print
                    print "Goal state (%i, %i) reached. Reward of %i obtained" % (
                           curr_state[0], curr_state[1], reward)
                    print
                          
                    total_rewards += reward

                    if(learn):
                        agent.update(prev_state, (selected_action, RoomRunner.actions), curr_state, reward)

                    agent.episode_over()
                    break
                else:
                    total_rewards += self.step_cost
                    
                    if(learn):
                        agent.update(prev_state, (selected_action, RoomRunner.actions), curr_state, self.step_cost)

            self.grid[curr_state[1], curr_state[0]] = RoomRunner.__empty_sym
                
        return (total_rewards, total_steps)

    def __execute_action(self, state, action):
        h = self.grid.shape[0]
        w = self.grid.shape[1]

        newstate = state.copy()
        if action == 'up':
            if(state[1] - 1 >= 0 and self.grid[state[1]-1, state[0]] != RoomRunner.__wall_sym):
                newstate[1] -= 1
        elif action == 'right':
            if(state[0] + 1 < w and self.grid[state[1], state[0]+1] != RoomRunner.__wall_sym):
                newstate[0] += 1
        elif action == 'down':
            if(state[1] + 1 < h and self.grid[state[1]+1, state[0]] != RoomRunner.__wall_sym):
                newstate[1] += 1
        elif action == 'left':
            if(state[0] - 1 >= 0 and self.grid[state[1], state[0]-1] != RoomRunner.__wall_sym):
                newstate[0] -= 1

        return newstate
            

if __name__ == "__main__":
    import sys, os

    if len(sys.argv) != 2:
        print "A filename is required"
        sys.exit(1)

    path = sys.argv[1]

    rr = RoomRunner(path)

    try:
        with open('agent.pickle', 'r') as f:
            print "Loading pickled agent"
            a = pickle.load(f)
    except IOError:
        print "No pickled agent found."
        print "Creating a new agent"
        a = QLearningRoomAgent(rr.grid.shape[0], rr.grid.shape[1], len(RoomRunner.actions))

    (total_rewards, total_steps) = rr.run_episodes(a, num_eps=1000, visualize=True)

    print "total rewards: %f" % total_rewards
    print "total steps: %i" % total_steps
    print "rewards per step: %f" % (float(total_rewards)/total_steps)

    #print "Q values for action %s" %  RoomRunner.actions[0]
    #print a.q_values[:,:,0].T
    #print "Q values for action %s" % RoomRunner.actions[1]
    #print a.q_values[:,:,1].T
    #print "Q values for action %s" % RoomRunner.actions[2]
    #print a.q_values[:,:,2].T
    #print "Q values for action %s" % RoomRunner.actions[3]
    #print a.q_values[:,:,3].T

    print "Agent epsilon value"
    print a.epsilon
    
    print
    print "Saving the agent"
    with open('agent.pickle', 'w') as f:
        pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)
    