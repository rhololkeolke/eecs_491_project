import numpy as np
import matplotlib.pyplot as plt
import yaml

from terminal import Color

import pdb

class RoomRunner(object):
    __wall_sym = '*'
    __empty_sym = '-'
    
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
                else:
                    if((i,j) in self.goals):
                        grid_rep += Color.green(' %s ' % elem)
                    else:
                        grid_rep += ' %s ' % elem
            grid_rep += '\n'
        return grid_rep

            

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print "A filename is required"
        sys.exit(1)

    path = sys.argv[1]

    rr = RoomRunner(path)

    print rr