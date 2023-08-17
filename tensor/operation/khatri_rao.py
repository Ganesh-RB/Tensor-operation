"""
File for the Khatri-Rao product.

"""

import numpy as np

def khatri_rao(A: np.ndarray, B: np.ndarray):
    """khatriRao computes the Khatri-Rao product of two matrices A and B.

    Args:
        A (np.ndarray): First matrix.
        B (np.ndarray): Second matrix.

    Returns:
        np.ndarray: Khatri-Rao product of A and B.

    """
    assert A.ndim == 2, "A must be a 2D matrix." 
    assert B.ndim == 2, "B must be a 2D matrix."

    # Check if the matrices have the same number of columns
    if A.shape[1] != B.shape[1]:
        raise ValueError("The number of columns of A and B must be the same.")

    # Compute the Khatri-Rao product
    prod = np.zeros((A.shape[0] * B.shape[0], A.shape[1]))
    for i in range(A.shape[1]):
        prod[:, i] = np.kron(A[:, i], B[:, i])

    return prod


def _get_product_index(I: np.array, Sz: np.array):
    """_get_product_index computes the indices of the Khatri-Rao product of a set of matrices.

    Args:
        I (np.array): Indices of the matrices.
        Sz (np.array): Size of the matrices.

    Returns:
        int : index in the Khatri-Rao product corresponding to the indices I of the matrices.  
    """
    assert I.ndim == 1, "I must be a 1D array."
    assert Sz.ndim == 1, "Sz must be a 1D array."
    assert I.shape[0] == Sz.shape[0], "I and Sz must have the same length."

    index = 0
    for i in range(I.shape[0]):
        index += np.prod(Sz[i + 1 :]) * I[i]

    return index

def _get_factor_indices(index: int, Sz: np.array):
    """_get_factor_indices computes the indices of the matrices corresponding to a given index in the Khatri-Rao product.

    Args:
        index (int): Index in the Khatri-Rao product.
        Sz (np.array): Size of the matrices.

    Returns:
        np.array: Indices of the matrices corresponding to the index in the Khatri-Rao product.
    """
    assert Sz.ndim == 1, "Sz must be a 1D array."

    I = np.zeros(Sz.shape[0], dtype=int)
    for i in range(Sz.shape[0]):
        I[i] = index // np.prod(Sz[i + 1 :])
        index = index % np.prod(Sz[i + 1 :])

    return I

def sampled_khatri_rao(S: np.array, factors: list):
    """sampled_khatri_rao computes the Khatri-Rao product of a set of matrices.

    Args:
        S (np.array): Sampled indices.
        factors (list): List of matrices.

    Returns:
        np.ndarray: Khatri-Rao product of the matrices.

    """
    # Check if the matrices have the same number of columns
    for i in range(len(factors) - 1):
        if factors[i].shape[1] != factors[i + 1].shape[1]:
            raise ValueError("The number of columns of the matrices must be the same.")

    # Compute the Khatri-Rao product
    prod = np.ones((S.shape[0], factors[0].shape[1]))
    for j, s in enumerate(S):
        indices = _get_factor_indices(s, np.array([A.shape[0] for A in factors]))
        for i in range(len(factors)):
            prod[j, :] *= factors[i][indices[i], :]

    return prod