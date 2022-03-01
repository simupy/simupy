import numpy as np
from scipy import interpolate


def callable_from_trajectory(t, curves, k=3):
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
        y=curves, x=t, k=k)

    return bspline

def trajectory_interpolant(res, for_="state", k=3, bc_type=None):
    """
    Construct an interpolating spline from a trajectory that handles potential
    discontinuities from events.

    Parameters
    ----------
    res : `SimulationResult` object
        The `SimulationResult` object that will be interpolated
    for_ : {"state" or "output"}
        Indicate whether to interpolate the state or output of the SimulationResult.

    Returns
    -------
    interpolated_callable : callable
        Callable which interpolates the given curve/trajectories
    """
    if for_=="state":
        vals = res.x
    elif for_=="output":
        vals = res.y
    else:
        raise ValueError("Unsupported `for_` value")

    event_idxs = np.where(np.diff(np.sign(res.e[:, -1]))!=0)[0]+1
    prev_idx = 0
    interps = []
    extra_val = k+1
    for event_idx in event_idxs:
        interps.append(interpolate.make_interp_spline(res.t[prev_idx:event_idx], vals[prev_idx:event_idx, :], k=k, bc_type=bc_type))
        prev_idx = event_idx
    if prev_idx < res.t.shape[0]:
        interps.append(interpolate.make_interp_spline(res.t[event_idx:], vals[event_idx:, :], k=k, bc_type=bc_type))

    ttl = []
    for interp in interps:
        if interp is interps[0]:
            ttl.append(interp.t[:-extra_val])
        elif interp is interps[-1]:
            ttl.append(interp.t[extra_val:])
        else:
            ttl.append(interp.t[extra_val:-extra_val])

    cc = np.concatenate([interp.c for interp in interps])
    tt = np.concatenate(ttl)
    return interpolate.BSpline.construct_fast(tt, cc, k)

def isclose(res1, res2, atol=1E-8, rtol=1E-5, mode='numpy', for_="output"):
    """
    Compare two trajectories

    Parameters
    ---------- 
    mode : {'numpy' or 'pep485'}
    """
    if for_=="state":
        shape1 = res1.x.shape[1]
        shape2 = res2.x.shape[1]
    elif for_=="output":
        shape1 = res1.y.shape[1]
        shape2 = res2.y.shape[1]
    else:
        raise ValueError("Unsupported `for_` value")

    if shape1 != shape2:
        raise ValueError("res1 and res2 %s should be the same dimension to compare" %
                         for_)

    eval_t_list = list(set(res1.t) | set(res2.t))
    eval_t_list.sort()
    eval_t = np.array(eval_t_list)

    eval_y1 = trajectory_interpolant(res1)(eval_t)
    eval_y2 = trajectory_interpolant(res2)(eval_t)

    if mode=='numpy':
        return np.all(np.isclose(eval_y1, eval_y2, rtol=rtol, atol=atol), axis=0)
    elif mode=='pep485':
        return np.all(np.abs(a - b) <= np.max(rtol*np.max(a, b), atol), axis=0)





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
    local_curves = np.array(curves).reshape(local_time.shape[0], -1).copy()
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
