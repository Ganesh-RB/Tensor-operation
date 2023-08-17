"""
File to matricize a given n order tensor.
"""

import numpy as np

def matricize(tensor, mode):
    """Matricize a given tensor along a given mode.

    Args:
        tensor (np.ndarray): Tensor to be matricized.
        mode (int): Mode along which the tensor is to be matricized.

    Returns:
        np.ndarray: Matricized tensor. i.e $X_{(mode)}$.

    """
    assert mode >= 0 and mode < tensor.ndim, "Mode must be between 0 and the tensor order."

    return np.reshape(np.moveaxis(tensor, mode, 0),
                      (tensor.shape[mode], -1), order='F')
