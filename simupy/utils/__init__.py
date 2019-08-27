import numpy as np
from scipy import interpolate


def callable_from_trajectory(t, curves):
    """
    Use scipy.interpolate.make_interp_spline to build cubic b-spline
    interpolating functions over a set of curves.

    Parameters
    ----------
    t : 1D array_like
        Array of m time indices of trajectory
    curves : 2D array_like
        Array of m x n vector samples at the time indices. First dimension
        indexes time, second dimension indexes vector components

    Returns
    -------
    interpolated_callable : callable
        Callable which interpolates the given curve/trajectories
    """
    bspline = interpolate.make_interp_spline(
        y=curves, x=t)

    return bspline


def discrete_callable_from_trajectory(t, curves):
    """
    Build a callable that interpolates a discrete-time curve by returning the
    value of the previous time-step.

    Parameters
    ----------
    t : 1D array_like
        Array of m time indices of trajectory
    curves : 2D array_like
        Array of m x n vector samples at the time indices. First dimension
        indexes time, second dimension indexes vector components

    Returns
    -------
    nearest_neighbor_callable : callable
        Callable which interpolates the given discrete-time curve/trajectories
    """
    local_time = np.array(t).copy()
    local_curves = np.array(curves).copy()
    def nearest_neighbor_callable(t, *args):
        return local_curves[
            np.argmax((local_time.reshape(1,-1)>=np.array([t]).reshape(-1,1)),
                axis=1), :]

    return nearest_neighbor_callable



def array_callable_from_vector_trajectory(tt, x, unraveled, raveled):
    """
    Convert a trajectory into an interpolating callable that returns a 2D
    array. The unraveled, raveled pair map how the array is filled in. See
    riccati_system example.

    Parameters
    ----------
    tt : 1D array_like
        Array of m time indices of trajectory
    xx : 2D array_like
        Array of m x n vector samples at the time indices. First dimension
        indexes time, second dimension indexes vector components
    unraveled : 1D array_like
        Array of n unique keys matching xx.
    raveled : 2D array_like
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
                    (slice(None), *iterator.multi_index),
                    vector_result[..., idx]
                )
            else:
                array_result[tuple(iterator.multi_index)] = vector_result[idx]
        return array_result

    return array_callable
