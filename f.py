import numpy as np

from numba import njit

@njit
def reconstruct(img, C, alpha, iteration_num):

    # Initialisation
    m, n = img.shape[:2]

    Phi = np.zeros((m, n, len(C)), dtype=np.float32)
    G = init_G(C, alpha)
    Q = init_Q(m, n, C, img)

    L = np.zeros((m, n, len(C)), dtype=np.float64)  # 64?
    U = np.zeros((m, n, len(C)), dtype=np.float64)

    R = init_R(m, n, C, Phi, Q, G)
    D = init_D(m, n, C, Phi, Q, G)

    # Recalculating
    for i in range(iteration_num):
        L, R, U, D, Phi = iteration(L, R, U, D, Phi, C, Q, G, m, n)

    # Reconstruction
    result = np.zeros((m, n), dtype=np.uint8)
    for i in range(0, m):
        for j in range(1, n - 1):
            result[i, j] = restore_k(i, j, C, L, R, U, D, Phi, Q)

    return result

# Init
def init_C(num):
    return np.rint( np.arange(0, 255, 255/num) )


def init_G(C, alpha):
    """
    >>> init_G(np.array([[24, 1],[123, 190]]), 2,)
    [[  0.,  -4.,  -8., -12.],
       [ -4.,   0.,  -4.,  -8.],
       [ -8.,  -4.,   0.,  -4.],
       [-12.,  -8.,  -4.,   0.]]
    """
    C_len = len(C)
    G = np.zeros( (C_len, C_len), dtype=np.float32 )
    for i in range(C_len):
        for j in range(C_len):
            G[i, j] = -abs(C[i] - C[j]) * alpha

    return G
@njit
def init_Q(m, n, C, img):
    """
    >>> init_Q(2,2,np.array([0,2]),np.array([[2,4],[0,5]]))
    [[[-2.,  0.],
        [-4., -2.]],

       [[ 0.,  0.],
        [-5., -3.]]]
    """
    C_len = len(C)
    Q = np.zeros( (m, n, C_len), dtype=np.float32 )

    for i in range(m):
        for j in range(n):
            for c in range(C_len):

                if img[i, j] != 0:
                    Q[i, j, c] = -abs(img[i, j] - C[c])
                else:
                    Q[i, j, c] = 0

    return Q

@njit
def init_R(m, n, C, Phi, Q, G):

    R = np.zeros((m, n, len(C)), dtype=np.float64)

    for i in range(m-2, -1, -1):
        for j in range(n-2, -1, -1):
            for c in range(len(C)):
                R[i, j, c] = recompute_R(i, j, c, R, Phi, C, Q, G)

    return R


@njit
def init_D(m, n, C, Phi, Q, G):

    D = np.zeros((m, n, len(C)), dtype=np.float64)

    for i in range(m-2, -1, -1):
        for j in range(n-2, -1, -1):
            for c in range(len(C)):
                D[i, j, c] = recompute_D(i, j, c, D, Phi, C, Q, G)

    return D


# L, R, U, D update
@njit
def recompute_L(i, j, k, R, Phi, C, Q, G):

    foo = np.zeros((len(C),), dtype=np.float32)
    for k_ in range(len(C)):
        foo[k_] = R[i, j-1, k_] + 0.5*Q[i, j-1, k_] - Phi[i, j-1, k_] +G[k_, k]

    return np.max(foo)


@njit
def recompute_R(i, j, k, R, Phi, C, Q, G):

    foo = np.zeros((len(C),), dtype=np.float32)
    for k_ in range(len(C)):
        foo[k_] = R[i, j+1, k_] + 0.5*Q[i, j+1, k_] + Phi[i, j+1, k_] +G[k_, k]

    return np.max(foo)


@njit
def recompute_U(i, j, k, R, Phi, C, Q, G):

    foo = np.zeros((len(C),), dtype=np.float32)
    for k_ in range(len(C)):
        foo[k_] = R[i-1, j, k_] + 0.5*Q[i-1, j, k_] - Phi[i-1, j, k_] +G[k_, k]

    return np.max(foo)


@njit
def recompute_D(i, j, k, R, Phi, C, Q, G):

    foo = np.zeros((len(C),), dtype=np.float32)
    for k_ in range(len(C)):
        foo[k_] = R[i+1, j, k_] + 0.5*Q[i+1, j, k_] + Phi[i+1, j, k_] + G[k_, k]

    return np.max(foo)


@njit
def iteration(L, R, U, D, Phi, C, Q, G, m, n):

    for i in range(1, m):
        for j in range(1, n):
            for c in range(len(C)):
                L[i, j, c] = recompute_L(i, j, c, L, Phi, C, Q, G)
                U[i, j, c] = recompute_U(i, j, c, U, Phi, C, Q, G)
                Phi[i, j, c] = (L[i, j, c] + U[i, j, c] - R[i, j, c] - D[i, j, c])/2
    # backward
    for i in range(m-2, -1, -1):
        for j in range(n-2, -1, -1):
            for c in range(len(C)):
                R[i, j, c] = recompute_R(i, j, c, R, Phi, C, Q, G)
                D[i, j, c] = recompute_D(i, j, c, D, Phi, C, Q, G)
                Phi[i, j, c] = (L[i, j, c] + U[i, j, c] - R[i, j, c] - D[i, j, c])/2

    return L, R, U, D, Phi


# Reconstruction
@njit
def restore_k(i, j, C, L, R, U, D, Phi, Q, ):
    foo = list()
    for k_ in range(len(C)):
        foo.append(L[i, j, k_] + R[i, j, k_] + Q[i, j, k_] - Phi[i, j, k_])
    return C[foo.index(max(foo))]
