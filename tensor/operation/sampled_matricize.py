import numpy as np
from .import index_manipulation as ind_man

def Sampled_Matricize(S :np.ndarray,tensor :np.ndarray,mode: int):
    assert isinstance(tensor, np.ndarray),"Wrong data type only numpy array accepted for S" 
    assert isinstance(S,np.ndarray),"Wrong data type only numpy array accepted for tensor"
    assert isinstance(mode,int),"Wrong data type ,only integer accepted for mode"
    assert mode>=0 and mode<tensor.ndim,"Invalid mode given"
    assert np.prod(tensor.shape) !=0,"Some mode is zero ,invalid tensor"
    assert S.ndim==1,"S is expected to be an 1d tensor"

    S_sz=len(S)
    N=tensor.ndim
    I_n=tensor.shape[mode]
    
    
    indices=np.zeros(shape=(I_n,S_sz,N),dtype=int)
    for i in range(I_n):
        for j in range(S_sz):
            indices[i,j,:]= ind_man.convert_to_tensor_index(tensor.shape,mode,(i,S[j]))

    ret=np.zeros(shape=(I_n*S_sz))
    indices=indices.reshape(-1,indices.shape[-1])
    for i in range(indices.shape[0]):
        ret[i]=tensor[tuple(indices[i,:])]
    ret=ret.reshape(I_n,S_sz)
    return ret

