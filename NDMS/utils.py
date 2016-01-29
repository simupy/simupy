import sympy as sp, numpy as np
from numpy.core.numeric import isscalar
from sympy.utilities.lambdify import implemented_function

sinc = implemented_function(sp.Function('sinc'), lambda x: np.sinc(x/np.pi) )

def process_vector_args(args):
    new_args = []
    for arg in args:
        if isinstance(arg,(sp.Matrix,np.ndarray)):
            shape = arg.shape
            if (min(shape) != 1 and len(shape) == 2) or len(shape) > 2:
                raise AttributeError("Arguments should only contain vectors")
            for i in range(max(shape)):
                if len(shape) == 1:
                    new_args.append(arg[i])
                elif shape[0] == 1:
                    new_args.append(arg[0,i])
                elif shape[1] == 1: 
                    new_args.append(arg[i,0])
        elif isinstance(arg,(list,tuple)):
            for element in arg:
                if isinstance(element,(list,tuple)):
                    raise AttributeError("Arguments should not be nested lists/tuples")
                new_args.append(element)
        else: # hope it's atomic!
            new_args.append(arg)

    return tuple(new_args)
    
def lambdify_with_vector_args(args, expr, modules=({'ImmutableMatrix': np.matrix}, "numpy", {"Mod": np.mod})):
    new_args = process_vector_args(args)
    
    # TODO: check what later verisons of SymPy need for modules/handling Mod  
    # TODO: apparently lambdify can't be trusted?? Eventually move to 
    # codeprinter? http://stackoverflow.com/a/10138307/854909
    f = sp.lambdify(new_args, expr, modules=modules)
    
    def lambda_function_with_vector_args(*func_args):
        new_func_args = process_vector_args(func_args)
        return f(*new_func_args)
    lambda_function_with_vector_args.__doc__ = f.__doc__
    return lambda_function_with_vector_args
