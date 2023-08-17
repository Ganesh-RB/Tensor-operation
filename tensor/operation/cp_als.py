"""
File for the Alternating Least Squares algorithm for CP decomposition.

Algorithm:
    CP Decomposition
    Input: Tensor X ∈ R^{J_1 × J_2 × ⋯ × J_N}, rank R
    Output: Factor matrices A^{(1)}, A^{(2)}, ⋯, A^{(N)}
    Initialize A^{(1)}, A^{(2)}, ⋯, A^{(N)} with random values
    While Not converged
        For n = 1 to N
            Compute A^{(n)^T} = (A^{(N)} ⊙ A^{(N-1)} ⊙ ⋯ ⊙ A^{(n+1)} ⊙ A^{(n-1)} ⊙ A^{(1)})^+ X^T_{(n)}
        EndFor
    EndWhile
"""

import numpy as np
from tensor.operation.kruskal import kruskal
from tensor.operation.khatri_rao import khatri_rao
from tensor.operation.matricize import matricize

def cpDecomposition(X: np.ndarray, rank: int, maxIter: int = 5, tol: float = 1e-6):
    """cpDecomposition performs CP decomposition of a tensor X using alternating least squares.

    Args:
        X (np.ndarray): Tensor to be decomposed.
        rank (int): Rank of the decomposition.
        maxIter (int, optional): Maximum number of iterations. Defaults to 1000.
        tol (float, optional): Tolerance for the stopping criterion. Defaults to 1e-6.

    Returns:
        np.ndarray: Factor matrices of the decomposition.

    """
    # Initialize factor matrices
    U = [np.random.rand(X.shape[i], rank) for i in range(X.ndim)]

    # Iterate until convergence
    for itr in range(maxIter):

        for i in range(X.ndim):

            khatriRaoProd = np.ones((1, rank))
            for j in range(X.ndim, 0, -1):
                if j != (i + 1):
                    khatriRaoProd = khatri_rao(khatriRaoProd, U[j - 1])

            U[i] = matricize(X, i) @ np.linalg.pinv(khatriRaoProd).T
        print("Iteration ", itr+1, " completed. loss =", np.linalg.norm(X - kruskal(*U)))

        # Check for convergence
        if np.linalg.norm(X - kruskal(*U)) < tol:
            break

    return np.array(U)



# np.reshape(np.moveaxis(tensor, mode, 0),(tensor.shape[mode], -1), order='F')
