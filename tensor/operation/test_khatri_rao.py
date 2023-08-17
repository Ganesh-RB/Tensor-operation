"""
Test file for khatri_rao.py
"""

import numpy as np
import pytest

from operation.khatri_rao import khatri_rao

def test_khatri_rao():
    """Test for the khatri_rao function.
    """
    # Define the test matrices
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8, 9], [10, 11, 12]])

    # Compute the Khatri-Rao product    
    C = khatri_rao(A, B)
    # print(C)

    # Check if the result is correct
    assert np.allclose(C, np.array([[7, 16, 27], [10, 22, 36], [28, 40, 54], [40, 55, 72]]))

    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    # Compute the Khatri-Rao product
    C = khatri_rao(A, B)
    # print(C)
    assert np.allclose(C, np.array([[7, 16, 27], [10, 22, 36], [13, 28, 45], [28, 40, 54], [40, 55, 72], [52, 70, 90]]))



def test_khatri_rao_error():
    """Test for the khatri_rao function.
    """
    # Define the test matrices
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    # Compute the Khatri-Rao product
    with pytest.raises(ValueError):
        C = khatri_rao(A, B)




test_khatri_rao()
# test_khatri_rao_error()