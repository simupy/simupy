import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from simupy.systems.symbolic import DynamicalSystem


def construct_explicit_matrix(name, n, m, symmetric=False, diagonal=0,
                              dynamic=False, **kwass):
    """
    construct a matrix of symbolic elements

    Parameters
    ----------
    name : string
        Base name for variables; each variable is name_ij, which
        admitedly only works clearly for n,m < 10
    n : int
        Number of rows
    m : int
        Number of columns
    symmetric : bool, optional
        Use to enforce a symmetric matrix (repeat symbols above/below diagonal)
    diagonal : bool, optional
        Zeros out off diagonals. Takes precedence over symmetry.
    dynamic : bool, optional
        Whether to use sympy.physics.mechanics dynamicsymbol. If False, use
        sp.symbols
    kwargs : dict
        remaining kwargs passed to symbol function

    Returns
    -------
    matrix : sympy Matrix
        The Matrix containing explicit symbolic elements
    """
    if dynamic:
        symbol_func = dynamicsymbols
    else:
        symbol_func = sp.symbols

    if n != m and (diagonal or symmetric):
        raise ValueError("Cannot make symmetric or diagonal if n != m")

    if diagonal:
        return sp.diag(
            *[symbol_func(
                name+'_{}{}'.format(i+1, i+1), **kwass) for i in range(m)])
    else:
        matrix = sp.Matrix([
            [symbol_func(name+'_{}{}'.format(j+1, i+1), **kwass)
             for i in range(m)] for j in range(n)
        ])

        if symmetric:
            for i in range(1, m):
                for j in range(i):
                    matrix[i, j] = matrix[j, i]
        return matrix


def matrix_subs(*subs):
    """
    Generate an object that can be passed into sp.subs from matrices, replacing
    each element in from_matrix with the corresponding element from to_matrix

    There are three ways to use this function, depending on the input:
    1. A single matrix-level subsitution - from_matrix, to_matrix
    2. A list or tuple of (from_matrix, to_matrix) 2-tuples
    3. A dictionary of {from_matrix: to_matrix} key-value pairs
    """
    # I guess checking symmetry would be better, this will do for now.
    if len(subs) == 2 and not isinstance(subs[0], (list, tuple, dict)):
        subs = [subs]
    if isinstance(subs, (list, tuple)):
        return tuple(
            (sub[0][i, j], sub[1][i, j])
            for sub in subs
            for i in range(sub[0].shape[0])
            for j in range(sub[0].shape[1]) if sub[0][i, j] != 0
        )
    elif isinstance(subs, dict):
        return {
            sub[0][i, j]: sub[1][i, j]
            for sub in subs.items()
            for i in range(sub[0].shape[0])
            for j in range(sub[0].shape[1]) if sub[0][i, j] != 0
        }


def block_matrix(blocks):
    """
    Construct a matrix where the elements are specified by the block structure
    by joining the blocks appropriately.

    Parameters
    ----------
    blocks : two level deep iterable of sympy Matrix objects
        The block specification of the matrices used to construct the block
        matrix.

    Returns
    -------
    matrix : sympy Matrix
        A matrix whose elements are the elements of the blocks with the
        specified block structure.
    """
    return sp.Matrix.col_join(
        *tuple(
            sp.Matrix.row_join(
                *tuple(mat for mat in row)) for row in blocks
        )
    )


def system_from_matrix_DE(mat_DE, mat_var, mat_input=None, constants={}):
    """
    Construct a symbolic DynamicalSystem using matrices. See
    riccati_system example.

    Parameters
    ----------
    mat_DE : sympy Matrix
        The matrix derivative expression (right hand side)
    mat_var : sympy Matrix
        The matrix state
    mat_input : list-like of input expressions, optional
        A list-like of input expressions in the matrix differential equation
    constants : dict, optional
        Dictionary of constants substitutions.

    Returns
    -------
    sys : DynamicalSystem
        A DynamicalSystem which can be used to numerically solve the matrix
        differential equation.
    """
    vec_var = list(set(sp.flatten(mat_var.tolist())))
    vec_DE = sp.Matrix.zeros(len(vec_var), 1)

    iterator = np.nditer(mat_DE, flags=['multi_index', 'refs_ok'])
    for it in iterator:
        i, j = iterator.multi_index
        idx = vec_var.index(mat_var[i, j])
        vec_DE[idx] = mat_DE[i, j]

    sys = DynamicalSystem(vec_DE, sp.Matrix(vec_var), mat_input,
                          constants_values=constants)
    return sys
