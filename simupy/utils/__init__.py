import numpy as np
import copy
from scipy import interpolate

def callable_from_trajectory(t, curves):
    """
    Use scipy.interpolate splprep to build cubic b-spline interpolating
    functions over a set of curves.
    
    Parameters
    ----------
    t : 1D array-like
        Array of m time indices of trajectory
    curves : 2D array-like
        Array of m x n vector samples at the time indices. First dimension
        indexes time, second dimension indexes vector components
    
    Returns
    -------
    interpolated_callable : callable
        Callable which interpolates the given curve/trajectories
    """
    tck_splprep = interpolate.splprep(
        x=[curves[:, i] for i in range(curves.shape[1])], u=t, s=0)

    def interpolated_callable(t, *args):
        return np.array(interpolate.splev(t, tck_splprep[0], der=0))

    return interpolated_callable

def array_callable_from_vector_trajectory(tt, x, unraveled, raveled):
    """
    Convert a trajectory into an interpolating callable that returns a 2D
    array. The unraveled, raveled map how the array is filled in. See
    riccati_system example.

    Parameters
    ----------
    tt : 1D array-like
        Array of m time indices of trajectory
    xx : 2D array-like
        Array of m x n vector samples at the time indices. First dimension
        indexes time, second dimension indexes vector components
    unraveled : 1D array-like
        Array of n unique keys matching xx.
    raveled : 2D array-like
        Array where the elements are the keys from unraveled. The mapping
        between unraveled and raveled is used to specify how the output array
        is filled in.

    Returns
    -------
    matrix_callable : callable
        The callable interpolating the trajectory with the specified shape. 
    """
    xn, xm = x.shape

    vector_callable = callable_from_trajectory(tt, x)
    if hasattr(unraveled, 'shape') and len(unraveled.shape) > 1:
        unraveled = np.array(unraveled).flatten().tolist()

    def array_callable(t):
        vector_result = vector_callable(t)
        as_array = False
        if isinstance(t, (list, tuple, np.ndarray)) and len(t) > 1:
            array_result = np.zeros((len(t),)+raveled.shape)
            as_array = True
        else:
            array_result = np.zeros(raveled.shape)

        iterator = np.nditer(raveled, flags=['multi_index', 'refs_ok'])
        for it in iterator:
            iterator.multi_index
            idx = unraveled.index(raveled[iterator.multi_index])
            if as_array:
                array_result.__setitem__(
                    (slice(None),*iterator.multi_index),
                    vector_result[idx]
                )
            else:
                array_result[tuple(iterator.multi_index)] = vector_result[idx]
        return array_result

    return array_callable
