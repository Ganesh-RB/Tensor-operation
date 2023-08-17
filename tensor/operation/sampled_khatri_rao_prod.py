import numpy as np
from .import index_manipulation as ind_man

def SKR(S :np.ndarray,factors,mode: int):
    """Finds sampled Khatri rao for a given range of indices

    Args:
        S (np.ndarray): the row numbers on which sampled khatri rao is to be found on
        factors (_type_): the factor matrices used for khatri rao calculation in standard formula
        mode (int): the mode for which this calculation is being done

    Returns:
        _type_: len(S)xR matrix of all the given rows form the actual khatri rao which were selected
    """
    # column no should be same
    assert all([x.ndim == 2 for x in factors]), "All matrices must be 2D."
    assert all(factors[0].shape[1]==x.shape[1] for x in factors), "All matrices must have the same number of columns."
    R=factors[0].shape[1]
    N=len(factors)
    S_sz=len(S)
    tensor_shape=[factors[i].shape[0] for i in range(len(factors))]

    indices=np.zeros(shape=(S_sz,N),dtype=int)
    
    for i in range(S_sz):
        indices[i,:]= ind_man.convert_to_tensor_index(tensor_shape,mode,(0,S[i]))
    ret=np.ones(shape=(S_sz,R))
    for n in range(N):
        if n==mode :
            continue
        # for tmp in range(S_sz):
        #     ret[tmp,:]=np.multiply(ret[tmp,:],factors[n][indices[tmp,n],:])
        ret*=factors[n][indices[:,n],:]
    return ret
