from typing import Union
import numpy as np
import torch

def calc_rbf_bandwidth(X: Union[np.ndarray, torch.Tensor]):
    if not isinstance(X, np.ndarray):
        X = X.cpu().numpy()
    n = X.shape[0]
    G = np.sum(X * X, 1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * np.dot(X, X.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    bandwidth = np.sqrt(0.5 * np.median(dists[dists > 0]))
    return bandwidth
    

def calc_rbf_gamma(X_tensor: torch.Tensor):
    bandwidth = calc_rbf_bandwidth(X_tensor)
    gamma = 0.5 * bandwidth**-2
    return gamma
    