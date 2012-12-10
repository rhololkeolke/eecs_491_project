import numpy as np
from numpy import linalg as LA
from lspiframework.policy import Policy

import pdb

def lstdq(samples, policy, new_policy):
    """
    Evaluates the policy using the samples.

    This is just the lstdq algorithm

    Returns the learned weights w and the
    matrices A and b of the linear system Aw=b
    """

    howmany = len(samples)
    k = new_policy.basis()
    A = np.zeros((k, k))
    b = np.zeros((k, 1))

    for i in range(howmany):
        phi = new_policy.basis(samples[i].state, samples[i].action)
        
        if not samples[i].absorb:
            nextaction = policy.select_action(samples[i].nextstate)[0]
            nextphi = new_policy.basis(samples[i].nextstate, nextaction)
        else:
            nextphi = np.zeros((k,1))

        # update the matrices A and b
        A = A + phi * (phi - new_policy.discount * nextphi).T
        b = b + phi * samples[i].reward

    # solve the system of equations to find w
    rankA = LA.matrix_rank(A)

    print "Rank of matrix A: %i" % rankA
    if rankA == k:
        print "A is a full rank matrix"
        w = LA.inv(A).dot(b)
    else:
        print "A is lower rank than %i" % k
        w = LA.pinv(A).dot(b)

    return w, A, b

def qvalue(state, action, policy):
    """
    Calculates the Q value for a state action pair
    given a policy
    """
    phi = policy.basis(state, action)
    qvalue = phi.T.dot(policy.weights)

    return qvalue
    
def lspi(maxiter, epsilon, samples, initial_policy):
    """
    Runs the LSPI algorithm
    """

    iteration = -1
    distance = float('inf')
    policy = initial_policy
    all_policies = [initial_policy]
    
    while (iteration < maxiter) and (distance > epsilon):

        # print the number of iterations
        iteration = iteration + 1
        print ('============================')
        print 'LSPI iteration: %i' % iteration
        if iteration == 0:
            firsttime = 1
        else:
            firsttime = 0

        policy = Policy.copy(policy)

        policy.weights = lstdq(samples, all_policies[iteration], policy)[0]

        diff = policy.weights - all_policies[iteration].weights
        LMAXnorm = LA.norm(diff, np.inf)
        L2norm = LA.norm(diff)

        distance = L2norm

        all_policies.append(policy)

    print '================================'
    if distance > epsilon:
        print 'LSPI finished in %i iterations WITHOUT convergence to a fixed point' % iteration
    else:
        print 'LSPI converged in %i iterations' % iteration
    print
    print 'weights'
    print policy.weights
    print

    return policy, all_policies        