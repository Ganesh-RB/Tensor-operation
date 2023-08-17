import numpy as np

def convert_to_mat_index(tensor_shape,mode,index):
    ret=np.zeros(2,dtype=int)
    ret[0]=index[mode]
    new_shape = tensor_shape[:mode] + tensor_shape[mode+1:]
    new_indices=index[:mode]+ index[mode+1:]
    for x in range(len(new_shape)):
        mul=1
        for y in range(x):
            mul*=new_shape[y]
        ret[1]+= mul*new_indices[x]
    return ret

def convert_to_tensor_index(tensor_shape,mode,matrix_index):
    N=len(tensor_shape)
    
    ret=np.zeros(N,dtype=int)
    ret[mode]=matrix_index[0]
    new_shape = tensor_shape[:mode] + tensor_shape[mode+1:]
    indices=tuple(range(N))
    indices=indices[:mode]+indices[mode+1:]
    
    tmp=matrix_index[1]
    for x in range(N-1):
        mul=1
        for y in range(x):
            mul*=new_shape[y]
        ret[indices[x]]=(tmp%(mul*new_shape[x]))//mul
    return ret
