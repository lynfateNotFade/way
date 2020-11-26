import numpy as np

def normalized_laplacian_matrix(S):
    D = np.diag(1 / np.sqrt(np.sum(S, axis=1) + 1e-15))
    return D @ S @ D