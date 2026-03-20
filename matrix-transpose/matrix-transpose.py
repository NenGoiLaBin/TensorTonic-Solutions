import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    N, M = A.shape
    result = np.empty((M,N))
    for i in range(N):
        for j in range(M):
            result[j][i] = A[i][j]
    return result
    pass
