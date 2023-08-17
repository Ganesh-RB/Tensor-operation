import numpy as np

def StoppingCriteria(original_tensor: np.ndarray, reconstructed_tensor: np.ndarray, P_hat: int, threshold=0.01) -> bool:
    '''
    This function is used to check if the tensor has converged.
    It randomly chooses P possible indices of original_tensor and new_tensor and compares the elements at those indices.
    '''
    # Randomly choose P indices
    og_ten = original_tensor.size
    indices = np.random.choice(og_ten, P_hat, replace=False)
    # print(original_tensor.size)
    # print elements at those indices
    # print(original_tensor.flat[indices])
    # print(indices)
    # print(new_tensor.flat[indices])
    
    # original_tensor_elements = original_tensor.flat[indices]
    # reconstructed_tensor_elements = reconstructed_tensor.flat[indices]
    original_tensor_elements = np.take(original_tensor, indices=indices)
    reconstructed_tensor_elements = np.take(reconstructed_tensor, indices=indices)
    
    # mean of squared difference
    # mean_squared_difference = np.sqrt(np.mean(np.square(original_tensor_elements - reconstructed_tensor_elements)) * og_ten)
    # print(type(original_tensor_elements))
    mean_squared_difference = np.linalg.norm(original_tensor_elements - reconstructed_tensor_elements) * np.sqrt(og_ten)
    
    # approx_epsilon = mean_squared_difference * original_tensor.size
    # approx_epsilon = approx_epsilon**(1/2)

    relative_residual = mean_squared_difference / np.linalg.norm(original_tensor)
    if relative_residual < threshold:
        return True
    return False
        
