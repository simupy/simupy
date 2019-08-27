import numpy as np
import sympy as sp
from sympy.utilities.lambdify import implemented_function
from sympy.physics.mechanics import dynamicsymbols
from simupy.array import r_, Array

DEFAULT_LAMBDIFY_MODULES = ({'ImmutableMatrix': np.matrix, "atan2": np.arctan2}, "numpy", {"Mod": np.mod, "atan2": np.arctan2})


def process_vector_args(args):
    """
    A helper function to process vector arguments so callables can take
    vectors or individual components. Essentially unravels the arguments.
    """
    new_args = []
    for arg in args:
        if hasattr(arg, 'shape') and len(arg.shape) > 0:
            shape = arg.shape
            if (min(shape) != 1 and len(shape) == 2) or len(shape) > 2:
                raise AttributeError("Arguments should only contain vectors")
            for i in range(max(shape)):
                if len(shape) == 1:
                    new_args.append(arg[i])
                elif shape[0] == 1:
                    new_args.append(arg[0, i])
                elif shape[1] == 1:
                    new_args.append(arg[i, 0])
        elif isinstance(arg, (list, tuple)):
            for element in arg:
                if isinstance(element, (list, tuple)):
                    raise AttributeError("Arguments should not be nested " +
                                         "lists/tuples")
                new_args.append(element)
        else:  # hope it's atomic!
            new_args.append(arg)

    return tuple(new_args)


def lambdify_with_vector_args(args, expr, modules=DEFAULT_LAMBDIFY_MODULES):
    """
    A wrapper around sympy's lambdify where process_vector_args is used so
    generated callable can take arguments as either vector or individual
    components

    Parameters
    ----------
    args : list-like of sympy symbols
        Input arguments to the expression to call
    expr : sympy expression
        Expression to turn into a callable for numeric evaluation
    modules : list
        See lambdify documentation; passed directly as modules keyword.

    """
    new_args = process_vector_args(args)

    if sp.__version__ < '1.1' and hasattr(expr, '__len__'):
        expr = sp.Matrix(expr)

    f = sp.lambdify(new_args, expr, modules=modules)

    def lambda_function_with_vector_args(*func_args):
        new_func_args = process_vector_args(func_args)
        return np.array(f(*new_func_args))
    lambda_function_with_vector_args.__doc__ = f.__doc__
    return lambda_function_with_vector_args


def grad(f, basis, for_numerical=True):
    """
    Compute the symbolic gradient of a vector-valued function with respect to a
    basis.

    Parameters
    ----------
    f : 1D array_like of sympy Expressions
        The vector-valued function to compute the gradient of.
    basis : 1D array_like of sympy symbols
        The basis symbols to compute the gradient with respect to.
    for_numerical : bool, optional
        A placeholder for the option of numerically computing the gradient.

    Returns
    -------
    grad : 2D array_like of sympy Expressions
        The symbolic gradient.
    """
    if hasattr(f, '__len__'):  # as of version 1.1.1, Array isn't supported
        f = sp.Matrix(f)

    return f.__class__([
        [
            sp.diff(f[x], basis[y])
            if not for_numerical or not f[x].has(sp.sign(basis[y])) else 0
            for y in range(len(basis))
        ] for x in range(len(f))
    ])


def augment_input(system, input_=[], update_outputs=True):
    """
    Augment input, useful to construct control-affine systems.

    Parameters
    ----------
    system : DynamicalSystem
        The sytsem to augment the input of
    input_ : array_like of symbols, optional
        The input to augment. Use to augment only a subset of input components.
    update_outputs : boolean
        If true and the system provides full state output, will also add the
        augmented inputs to the output.
    """
    # accept list, etc of symbols to augment
    augmented_system = system.copy()
    if input_ == []:
        # augment all
        input_ = system.input

    augmented_system.state = r_[system.state, input_]
    augmented_system.input = Array([
        dynamicsymbols(str(input_var.func) + 'prime')
        for input_var in input_
    ])
    augmented_system.state_equation = r_[
        system.state_equation, augmented_system.input]

    if update_outputs and system.output_equation == system.state:
        augmented_system.output_equation = augmented_system.state

    return augmented_system
