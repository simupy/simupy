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

def trajectory_interpolant(res, for_="output", cols=slice(None), k=3, bc_type="natural"):
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
        vals = res.x[:, cols]
    elif for_=="output":
        vals = res.y[:, cols]
    else:
        raise ValueError("Unsupported `for_` value")

    event_idxs = np.unique(np.where(np.diff(np.sign(res.e), axis=0)!=0)[0])+1
    if event_idxs.size > 0 and event_idxs[0] == 1:
        event_idxs = event_idxs[1:]
    prev_idx = 0
    interps = []
    extra_val = int((k+1)//2)
    event_idx = 0
    for event_idx in event_idxs:
        interps.append(
            interpolate.make_interp_spline(
                res.t[prev_idx:event_idx],
                vals[prev_idx:event_idx, :],
                k=k, bc_type=bc_type
            )
        )
        prev_idx = event_idx
    if prev_idx < res.t.shape[0]:
        interps.append(
            interpolate.make_interp_spline(
                res.t[event_idx:],
                vals[event_idx:, :],
                k=k, bc_type=bc_type
            )
        )

    ccl = []
    for interp in interps:
        if interp is not interps[0]:
            ccl.append(interp.c[[0]*extra_val, :])

        ccl.append(interp.c)

        if interp is not interps[-1]:
            ccl.append(interp.c[[-1]*extra_val, :])

    cc = np.concatenate(ccl)
    tt = np.concatenate([interp.t for interp in interps])
    return interpolate.BSpline.construct_fast(tt, cc, k)

def trajectory_linear_combination(res1, res2, coeff1=1, coeff2=1, for_="output", cols=slice(None), k=3, bc_type="natural"):

    if for_=="state":
        shape1 = res1.x.shape[1]
        shape2 = res2.x.shape[1]
    elif for_=="output":
        shape1 = res1.y.shape[1]
        shape2 = res2.y.shape[1]
    else:
        raise ValueError("Unsupported `for_` value")

    eval_t_list = list(set(res1.t) | set(res2.t))
    eval_t_list.sort()
    eval_t = np.array(eval_t_list)

    interp1 = trajectory_interpolant(res1, for_, cols, k, bc_type)
    interp2 = trajectory_interpolant(res2, for_, cols, k, bc_type)

    for t in eval_t:
        if t not in interp1.t:
            interp1 = interpolate.insert(t, interp1)
        if t not in interp2.t:
            interp2 = interpolate.insert(t, interp2)

    if not np.all(interp1.t == interp2.t):
        raise ValueError("Expected to construct the same knot structure for each interpolant")

    if not np.all(np.unique(interp1.t) == eval_t):
        raise ValueError("Expected the knots to lie on eval_t")

    return interpolate.BSpline.construct_fast(interp1.t,
                                              interp1.c*coeff1 + interp2.c*coeff2,
                                              k)

def trajectory_norm(interp, p=2):
    eval_t = np.unique(interp.t)
    if p == np.inf:
        return np.max(np.abs(interp(eval_t)), axis=0)
    if p == -np.inf:
        return np.min(np.abs(interp(eval_t)), axis=0)
    if isinstance(p, int):
        integrand = interpolate.BSpline.construct_fast(interp.t, np.abs(interp.c)**p,
                                                       interp.k)
        return (integrand.antiderivative()(eval_t[-1])**(1/p))/np.diff(eval_t[[0,-1]])
    raise ValueError("unexpected value for p")


def isclose(res1, res2, p=np.Inf, atol=1E-8, rtol=1E-5, mode='numpy', for_="output",
            cols=slice(None), k=3, bc_type="natural"):
    """
    Compare two trajectories

    Parameters
    ---------- 
    mode : {'numpy' or 'pep485'}
    """
    traj_diff = trajectory_linear_combination(res1, res2, coeff1=1, coeff2=-1,
                                              for_=for_, cols=cols, k=k, bc_type=bc_type)
    diff_norm = trajectory_norm(traj_diff, p)
    res1_norm = trajectory_norm(trajectory_interpolant(res1, for_=for_, cols=cols, k=k,
                                                       bc_type=bc_type), p)
    res2_norm = trajectory_norm(trajectory_interpolant(res2, for_=for_, cols=cols, k=k,
                                                       bc_type=bc_type), p)
    if mode=='numpy':
        return (diff_norm <= (atol + rtol*res2_norm))
    elif mode=='pep485':
        return (diff_norm <= np.clip(
                    rtol*np.max(np.stack([res1_norm, res2_norm]), axis=0),
                    a_min=atol, a_max=None)
                )





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
