import numpy as np


def signature(path, level):
    if level == 0:  # constant
        return 1
    if level == 1:  # vector
        return path[:, -1]-path[:, 0]
    if level == 2:  # matrix
        N = path.shape[0]
        S = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    S[i, j] = np.sum((path[i]-path[i, 0])[1:]*(path[j, 1:]-path[j, :-1]))

        return S
    else:
        raise NotImplementedError


def lead_matrix(path, S=None):
    N = path.shape[0]
    A = np.zeros((N, N))

    if S is None:  # use explicit interpolation construction
        for i in range(N - 1):
            for j in range(i + 1, N):
                A[i, j] = path[j, 0]*path[i, -1] - path[i, 0]*path[j, -1] + np.sum(path[i, :-1]*path[j, 1:] - path[j, :-1]*path[i, 1:])

        A = 0.5 * (A - A.T)
    else:  # use A[i,j] = 0.5 * (S[i,j] - S[j,i])
        A = 0.5 * (S - S.T)

    return A
