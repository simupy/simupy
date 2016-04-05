import sympy as sp, numpy as np
from numpy.core.numeric import isscalar
from sympy.utilities.lambdify import implemented_function
from scipy import interpolate
#from .Systems

sinc = implemented_function(sp.Function('sinc'), lambda x: np.sinc(x/np.pi) )

def process_vector_args(args):
    # TODO: any combination of vectors for each arg should be allowed
    # TODO: be more vectorizable? allow vectors for each arg
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
    # TODO: should this figure out how to use u-funcify if possible?
    # TODO: be more vectorizable? each symbol can be a list/vector, return list/vectors
    f = sp.lambdify(new_args, expr, modules=modules)
    
    def lambda_function_with_vector_args(*func_args):
        new_func_args = process_vector_args(func_args)
        return f(*new_func_args)
    lambda_function_with_vector_args.__doc__ = f.__doc__
    return lambda_function_with_vector_args

def callable_from_trajectory(t,curves):
    # TODO: Could write pre-allow passing pre-/post- processing functions??
    # Is there a better design for guessing how everything is split up??
    # let's make it be concatenated
    tck_splprep = interpolate.splprep(x=[curves[:,i] for i in range(curves.shape[1]) ], u=t, s=0)
    def interpolated_callable(t,*args):
        return interpolate.splev(t, tck_splprep[0], der=0)
    return interpolated_callable

def grad(f, basis):
    return sp.Matrix([ 
        [ sp.diff(f[x],basis[y]) for y in range(len(basis)) ] \
            for x in range(len(f)) ])

def augment_inputs(system):
    # Augment inputs, useful to construct control-affine systems
    if isinstance(system, TSFM) and system.dt:
        def new_call_function(self, x, u):
            return N.append(self.old_callable_function(x[:self.n_states-self.n_inputs], x[-self.n_inputs:] + u), x[-self.n_inputs:] + u, 0)
    elif isinstance(system, TSFM):
        def new_call_function(self, x, u):
            return N.append(self.old_callable_function(x), u, 0)
    elif isinstance(system, NonlinearSystem) and system.dt:
        def new_call_function(self, x, u):
            return N.append(self.old_callable_function(x[:self.n_states-self.n_inputs], x[-self.n_inputs:] + u), x[-self.n_inputs:] + u, 0)
    elif isinstance(system, NonlinearSystem):
        def new_call_function(self, x, u):
            return N.append(self.old_callable_function(x), u, 0)
    else:
        raise ValueError
    
    system = system.copy()
    system.old_callable_function = system.callable_function
    system.callable_function = types.MethodType(new_call_function, system)
    system.n_states = system.n_inputs + system.n_states
    return system
        