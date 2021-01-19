import numpy as np
from scipy.spatial.distance import pdist, squareform


def diffusion_distance(W, tau, p):
    """calculate the diffusion distance for scale tau and raise to the power of p"""
    print("diffusion distance")
    N = W.shape[0]
    d = np.sum(W, axis=0)  # degrees
    # initialize the Markov Matrix
    P_1 = W / d[:, np.newaxis]
    connected = False
    if np.allclose(np.sum(P_1, axis=1), np.ones(N)):
        connected = True
    P = P_1
    for t in range(tau - 1):
        P_1 = P_1 @ P
    pi_y = d / np.sum(d)
    distances = pdist(P_1, 'seuclidean', V=pi_y)
    Dt = squareform(distances)
    C = np.power(Dt, p)
    C = C + np.eye(N) * 1e-8
    # C = C / np.max(C)  # normalize cost to avoid numerical issues - recommended for larger graphs
    return C, connected


