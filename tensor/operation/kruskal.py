import numpy as np
import string

def nary_outer_einsum_52(*vectors):
    subscripts = string.ascii_letters[:len(vectors)]
    subscripts = ','.join(subscripts) + '->' + subscripts
    return np.einsum(subscripts, *vectors)

def kruskal(*args):
    """kruskal computes the outer product of a list of matrices and sums them.
    Args:
        *args (np.ndarray): List of matrices.
    Returns:
        np.ndarray: kronical product with identity tensor
    """
    assert len(args) > 1, "At least two matrices are required."
    assert all([A.ndim == 2 for A in args]), "All matrices must be 2D."
    assert all([A.shape[1] == args[0].shape[1] for A in args]), "All matrices must have the same number of columns."

    R = args[0].shape[1]
    out = np.zeros(tuple([A.shape[0] for A in args]))
    for i in range(R):
    #     out += np.prod(np.ix_(*[A[:, i] for A in args]))
        out+=nary_outer_einsum_52(*[A[:, i] for A in args])

    return out

