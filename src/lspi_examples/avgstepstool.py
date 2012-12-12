import rooms
from lspiframework.policy import Policy, RandomPolicy, ValuePolicy
from lspiframework import lspi
from protovalueframework import pvf
import numpy as np
import matplotlib.pyplot as plt
import ast
import pdb

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', help='A list representing the range of \
                        episodes to use for training. e.g. [0, 100, 20] says to \
                        run 0 to 100 episodes in steps of size 20')
    parser.add_argument('--max-steps', help='Maximum number of steps per episode', type=int)
    parser.add_argument('-k', help='A list represeting the range of pvf functions \
                        to use. e.g. [25 75 5] says to run this test with policies using \
                        basis functions constructed of 25 to 75 pvfs in 5 pvf steps')
    parser.add_argument('-n', help='The number of iterations of execution to test this \
                        with', type=int)
    parser.add_argument('-g', help='The graph output file name')
    parser.add_argument('-o', help='Data file name')
    parser.add_argument('--map', help='The map file name')
    args = parser.parse_args()

    # deal with the episode parameter
    if args.episodes:
        temp = ast.literal_eval(args.episodes)
        start_episode = temp[0]
        end_episode = temp[1]

        if len(temp) == 3:
            step_episode = temp[2]
        else:
            step_episode = 10
    else:
        start_episode = 0
        end_episode = 100
        step_episode = 10

    if args.max_steps:
        max_steps = args.max_steps
    else:
        max_steps = 100

    if args.k:
        temp = ast.literal_eval(args.k)
        start_k = temp[0]
        end_k = temp[1]

        if len(temp) == 3:
            step_k = temp[2]
        else:
            step_k = 25
    else:
        start_k = 25
        end_k = 100
        step_k = 25

    if args.n:
        num_tries = args.n
    else:
        num_tries = 10

    if args.g:
        graph_output = args.g
    else:
        graph_output = 'average_steps_tool_graph.png'

    if args.o:
        data_output = args.o
    else:
        data_output = 'average_steps_tool_data.csv'

    try:
        data_file = open(data_output, 'w')
    except IOError:
        print "Error opening data file"
        data_file.close()
    
    if args.map:
        room_config = args.map
    else:
        room_config = 'config.yaml'

    maxiter = 20
    epsilon = 10**-5
    discount = .8

    sim = rooms.Simulator(room_config)

    data = {}
    final_data = {}
    for episode in range(start_episode, end_episode+1, step_episode):
        for k in range(start_k, end_k+1, step_k):
            print "episode: %i k: %i" % (episode, k)

            if (k, episode) not in data:
                data[(k, episode)] = []
            samples = rooms.collect_samples(sim, maxepisodes=episode, maxsteps=max_steps)

            graph = pvf.construct_graph(samples, sim.states)
            try:
                basis = pvf.create_basis_function(graph, sim.states,
                                                      sim.actions, k)
            except:
                print "Couldn't compute basis function for this data"
                continue
                    
            policy = rooms.initialize_policy(0.0, discount, basis)

            final_policy = lspi.lspi(maxiter, epsilon,
                                         samples, policy)[0]

            for n in range(num_tries):                
                execution_data = rooms.test_execution(sim, final_policy, maxsteps=max_steps)
                

                data[(k, episode)].append(execution_data)

    for episode in range(start_episode, end_episode+1, step_episode):
        for k in range(start_k, end_k+1, step_k):
            total_steps = 0
            data_list = data.get((k, episode), [])
            for data_point in data_list:
                total_steps += data_point[2]

            if k not in final_data:
                final_data[k] = []
            if len(data_list) != 0:
                final_data[k].append(total_steps/len(data_list))
            else:
                final_data[k].append(0)


    plt.figure()
    plt.hold(True)
    for k in final_data:
        plt.plot(np.arange(start_episode, end_episode+1, step_episode), final_data[k])

    legend = []
    for k in final_data:
        legend.append('%i basis functions' % k)

    plt.legend(legend)
    plt.title('Comparison of PVF performance for map %s' % room_config)
    plt.xlabel('Number of training episodes (each episode max length of %i)' % max_steps)
    plt.ylabel('Average Number of Steps over %i runs' % num_tries)

    plt.savefig(graph_output)
    plt.show()

    data_file.write('k, data\n')
    for k in final_data:
        data_file.write('%i,' % k)
        for data_point in final_data[k]:
            data_file.write('%f,' % data_point)
        data_file.write('\n')

    # compute the Q-value error

    data_file.close()
