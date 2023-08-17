import numpy as np

def right_pseudo_inverse(A):
    """Returns the right pseudo inverse of a matrix.
       Returns: A+ = A^T (A A^T)^-1
       Shape: (n,m) -> (m,n)
    """
    return A.T @ np.linalg.inv(A @ A.T)