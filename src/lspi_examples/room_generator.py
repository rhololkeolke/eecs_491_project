import yaml
import random

class MazeGen(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width

        self.unvisited = {}
        
        for row in range(0, height, 2):
            for col in range(0, width, 2):
                self.unvisited[(row, col)] = True

        self.stack = []
        self.doors = []

        for key in self.unvisited:
            self.doors.append(key)

    def select_unvisited_neighbor(self, state):
        r,c = state
        moves = []
        if r-2 >= 0 : moves.append((r-2,c))
        if c-2 >= 0 : moves.append((r,c-2))
        if r+2 < self.height : moves.append((r+2,c))
        if c+2 < self.width : moves.append((r,c+2))

        neighbors = []
        for move in moves:
            if self.unvisited.get(move, None) is not None:
                neighbors.append(move)

        if len(neighbors) == 0:
            return None
        return random.choice(neighbors)

    def generate(self):
        curr_cell = (0,0)
        if curr_cell in self.unvisited:
            del self.unvisited[curr_cell]

        while len(self.unvisited) != 0:
            next_cell = self.select_unvisited_neighbor(curr_cell)

            if next_cell is not None:
                self.stack.append(curr_cell)
                self.doors.append((curr_cell[0]+(next_cell[0]-curr_cell[0])/2,
                                   curr_cell[1]+(next_cell[1]-curr_cell[1])/2))
                curr_cell = next_cell
                del self.unvisited[curr_cell]
            elif len(self.stack) != 0:
                curr_cell = self.stack[-1]
                del self.stack[-1]
            else:
                curr_cell = random.choice(self.unvisited.keys())
                del self.unvisited[curr_cell]

    def write_file(self, path, action_set, step_cost, succprob, num_goals, reward):
        config = {}
        config['actions'] = action_set
        config['step cost'] = step_cost
        config['succprob'] = succprob
        config['height'] = self.height
        config['width'] = self.width

        walls = {}
        horizontal = []
        # fill the entire grid with walls
        for i in range(self.height):
            horizontal.append(i)

        walls['horizontal'] = horizontal
        config['walls'] = walls

        doors = []
        # insert the doors
        for door in self.doors:
            doors.append({'x': door[1], 'y': door[0]})

        config['doors'] = doors

        # generate goals
        goals = []
        for i in range(num_goals):
            goals.append(random.choice(self.doors))
        
        absorbing = []
        # insert the goals
        for goal in goals:
            absorbing.append({'x': goal[1], 'y': goal[0], 'r': reward})

        config['absorbing'] = absorbing
        
        with open(path, 'w') as f:
            f.write(yaml.dump(config))




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, help='The height of the map')
    parser.add_argument('-w', type=int, help='The width of the map')
    parser.add_argument('-a', help='Action set (manhattan|diagonal)')
    parser.add_argument('-s', type=float, help='step cost')
    parser.add_argument('-p', type=float, help='success probability')
    parser.add_argument('-g', type=int, help='number of goals')
    parser.add_argument('-r', type=float, help='reward amount for each goal')
    parser.add_argument('-o', help='The output file name')
    
    args = parser.parse_args()

    if args.height is None:
        args.height = 10
    if args.w is None:
        args.w = 10
    if args.a is None:
        args.a = 'manhattan'
    if args.s is None:
        args.s = 0.0
    if args.p is None:
        args.p = 0.9
    if args.o is None:
        args.o = 'maze.yaml'
    if args.g is None:
        args.g = 1
    if args.r is None:
        args.r = 100.0
    
    m = MazeGen(args.height, args.w)
    m.generate()
    m.write_file(args.o, args.a, args.s, args.p, args.g, args.r)
