from apgl.graph import SparseGraph
from scipy.sparse.linalg import eigsh
import numpy as np

def construct_graph(samples, S):
    """
    Constructs an undirected graph using the
    Another Python Graph Library (apgl) SparseGraph
    class.

    samples are the samples used to construct the graph
    S is the number of states

    This method returns the constructed graph object
    """

    graph = SparseGraph(S)

    # loop through all samples
    # for each state transition
    # make the adjacency cell 1
    for sample in samples:
        if graph[sample.state, sample.nextstate] != 1:
            graph.addEdge(sample.state, sample.nextstate)

    return graph

def create_basis_function(graph, S, A, k):
    """
    This method computes the basis functions
    by taking the k smoothest eigen vectors
    of the combinatorial laplacian
    """

    laplacian = graph.normalisedLaplacianSym()
    
    eigen_vals, eigen_vecs =  eigsh(laplacian,
                                    k=k,
                                    which='SM')

    def basis(state=None, action=None):
        """
        Computes a set of polynomial (on "state") basis functions
        up to a certain degree. The set is duplicated for each action.
        The action determines which segement will be active
        """

        numbasis = A*k

        if state is None or action is None:
            return numbasis

        # initialize
        phi = np.zeros((numbasis, 1))
        
        # check if state is within bounds
        if state < 0 or state >= S:
            raise IndexError('%i is out of bounds' % state)

        # find the starting position
        base = action * (numbasis/A)

        phi[base:base+k, 0] = eigen_vecs[state, :].T

        return phi

    return basis