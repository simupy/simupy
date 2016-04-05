import numpy as np, sympy as sp, numpy.matlib
from sympy.physics.mechanics import dynamicsymbols
from .Systems import DynamicalSystem
from .utils import callable_from_trajectory

def construct_explicit_matrix(name, n, m, symmetric=False, dynamic=False, **kwass):
    if dynamic:
        symbol_func = dynamicsymbols
    else:
        symbol_func = sp.symbols
    
    matrix = sp.Matrix([ [ symbol_func(name+'_%d%d'%(j+1,i+1),**kwass) for i in range(m)] for j in range(n) ])
    if not symmetric:
        return matrix
    elif n!=m:
        raise ValueError("Cannot make symmetric if n != m")
    else:
        for i in range(1,m):
            for j in range(i):
                matrix[i,j] = matrix[j,i]
        return matrix

def matrix_subs(subs):
    # I guess checking symmetry would be better, this will do for now.
    if isinstance(subs,(list,tuple)):
        return tuple( (sub[0][i,j], sub[1][i,j]) for sub in subs for i in range(sub[0].shape[0]) for j in range(sub[0].shape[1]))
    elif isinstance(subs,dict):
        return { sub[0][i,j]: sub[1][i,j] for sub in subs.items() for i in range(sub[0].shape[0]) for j in range(sub[0].shape[1]) }
    
def matrix_callable_from_vector_trajectory(t,x,unraveled,raveled):
    xn,xm = x.shape
    if xm == t.size:
        time_axis = 1
        data_axis = 0
    else:
        time_axis = 0
        data_axis = 1
        
    vector_callable = callable_from_trajectory(t,x)
    if isinstance(unraveled,sp.Matrix):
        unraveled = sp.flatten(unraveled.tolist())
    def matrix_callable(t):
        vector_result = vector_callable(t)
        as_array = False
        if isinstance(t,(list,tuple,np.ndarray)) and len(t)>1:
            matrix_result = np.zeros(raveled.shape+(len(t),))
            as_array = True
        else:
            matrix_result = np.matlib.zeros(raveled.shape)
        
        iterator = np.nditer(raveled, flags=['multi_index','refs_ok'])
        for it in iterator:
            i,j = iterator.multi_index
            idx = unraveled.index(raveled[i,j])
            if as_array:
                matrix_result[i,j,:] = vector_result[idx]
            else:
                matrix_result[i,j] = vector_result[idx]
        
        return matrix_result
    return matrix_callable
            
def system_from_matrix_DE(mat_DE, mat_var, mat_input=sp.Matrix([]), subs=[]):
    # Sorry, not going to be clever and allow sets of DEs and variable matrices
    vec_var = list(set(sp.flatten(mat_var.tolist())))
    vec_DE = sp.Matrix.zeros(len(vec_var),1)
    
    iterator = np.nditer(mat_DE, flags=['multi_index','refs_ok'])
    for it in iterator:
        i,j = iterator.multi_index
        idx = vec_var.index(mat_var[i,j])
        vec_DE[idx] = mat_DE[i,j]
        
    sys = DynamicalSystem(vec_DE, sp.Matrix(vec_var), mat_input, constants_values=matrix_subs(subs))
    return sys