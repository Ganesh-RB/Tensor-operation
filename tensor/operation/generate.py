import numpy as np
import matplotlib.pyplot as plt
import wget
import os
from tensor.operation.kruskal import kruskal

def random_tensor(mode:int, shape, rank:int, noise:float =0):
    """Generate a random tensor with given shape and rank.
    Args:
        mode (int): The mode of the tensor.
        shape (tuple): The shape of the tensor.
        rank (int): The rank of the tensor.
        noise (float): The noise level of the tensor.

    Returns:
        Tensor: The generated tensor given by: $X = X_{true} + noise * \frac{\|X_{true}\|}{\|N\|} * N$,

    Example:
        >>> X = random_tensor(3, (10, 10, 8), 5, 0.1)
        >>> X.shape
        (10, 10, 8) 
    """
    # convert tuple to list
    N = np.random.rand(*shape)
    assert mode == len(shape), "Length of shape must be equal to mode."

    # generate a random matrices
    matrices = [np.random.rand(s, rank) for s in shape]
    X_true = kruskal(*matrices)
    X = X_true + noise * (np.linalg.norm(X_true)/np.linalg.norm(N)) * N
    return X 

def coil100_data(n:int = 100):
    """Coil-100 dataset. with 100 with 72 different angle images of size 128x128x3
    Args:
        n (int): The number of images to be returned. Default is 100.

    Returns:
        Tensor: The coil-100 dataset.
    """
    url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("tensor/operation", "data")
    os.makedirs(directory, exist_ok=True)

    # check if the file is already downloaded
    if not os.path.exists(directory + "/coil-100"):
        filename = wget.download(url, out=directory)    
        os.system("unzip " + filename + " -d " + directory)
        os.system("rm " + filename)
    
    # load the data
    N =np.random.choice(100, n, replace=False)
    X = np.zeros((128, 128, 3, n*72))
    for i, n in enumerate(N):
        for j in range(72):
            X[:, :, :, i*72 + j] = plt.imread(directory + "/coil-100/obj" + str(n+1) + "__" + str(5*j) + ".png")    

    return X

    
