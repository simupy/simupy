import pytest
import numpy as np
from scipy import linalg, signal
from simupy.systems import LTISystem, SystemFromCallable
import simupy.block_diagram as block_diagram

BlockDiagram = block_diagram.BlockDiagram
block_diagram.DEFAULT_INTEGRATOR_OPTIONS['rtol'] = 1E-9


def get_double_integrator(m=1000, b=50, d=1):
    N = 2
    sys = LTISystem(
        np.c_[[0, 1], [1, -b/m]],  # A
        np.r_[0, d/m],  # B
        # np.r_[0, 1],  # C
    )

    def ref_func(*args):
        if len(args) == 1:
            x = np.zeros(N)
        else:
            x = args[1]
        return np.r_[d/m, 0]-x
    ref = SystemFromCallable(ref_func, N, N)

    return sys, ref


def get_electromechanical(b=1, R=1, L=1, K=np.pi/5, M=1):
    # TODO: determine good reference and/or initial_condition
    # TODO: determine good default values for b, R, L, M
    N = 3
    sys = LTISystem(
        np.c_[  # A
            [0, 0, 0],
            [1, -b/M, -K/L],
            [0, K/M, -R/L]
        ],
        np.r_[0, 0, 1/L],  # B
        # np.r_[1, 0, 0],  # C
    )
    sys.initial_condition = np.ones(N)

    def ref_func(*args):
        if len(args) == 1:
            x = np.zeros(N)
        else:
            x = args[1]
        return np.r_[0, 0, 0]-x
    ref = SystemFromCallable(ref_func, N, N)

    return sys, ref


def get_double_mechanical(m1=1, k1=1, b1=1, m2=1, k2=0, b2=0):
    # TODO: determine good default values for m1, k1, b1, m2?
    N = 4
    sys = LTISystem(
        np.c_[  # A
            [0, -k1/m1, 0, k1/m2],
            [1, -b1/m1, 0, b1/m2],
            [0, k1/m1, 0, -(k1+k2)/m2],
            [0, b1/m1, 1, -(b1+b2)/m2]
        ],
        np.r_[0, 1/m1, 0, 0],  # B
        # np.r_[0, 0, 1, 0],  # C
    )
    sys.initial_condition = np.r_[
        0.5/k1 if k1 else 0,
        0,
        0.5/k2 if k2 else 0,
        0
    ]

    def ref_func(*args):
        if len(args) == 1:
            x = np.zeros(N)
        else:
            x = args[1]
        return np.zeros(N)-x
    ref = SystemFromCallable(ref_func, N, N)

    return sys, ref


def get_cart_pendulum(m=1, M=3, L=0.5, g=9.81, pedant=False):
    N = 4
    sys = LTISystem(
        np.c_[  # A
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, m*g/M, 0, (-1)**(pedant)*(m+M)*g/(M*L)],
            [0, 0, 1, 0]
        ],
        np.r_[0, 1/M, 0, 1/(M*L)],  # B
        # np.r_[1, 0, 1, 0]  # C
    )
    sys.initial_condition = np.r_[0, 0, np.pi/3, 0]

    def ref_func(*args):
        if len(args) == 1:
            x = np.zeros(N)
        else:
            x = args[1]
        return np.zeros(N)-x
    ref = SystemFromCallable(ref_func, N, N)

    return sys, ref


@pytest.fixture(
    scope="module",
    params=[
        get_electromechanical(),  # generic electromechanical

        get_double_mechanical(),  # free end
        get_double_mechanical(k2=1, b2=1),  # sprung-damped end

        get_cart_pendulum(),  # generic cart-pendulum model

        get_double_integrator(m=1, d=1, b=4.0),  # overdamped
        get_double_integrator(m=1, d=1, b=2.0),  # critically damped
        get_double_integrator(m=1, d=1, b=1.0),  # underdamped
        get_double_integrator(m=1, d=1, b=0.0),  # not damped
        get_double_integrator(m=1, d=1, b=-0.5),  # unstable
    ]
)
def control_systems(request):
    ct_sys, ref = request.param
    Ac, Bc, Cc = ct_sys.data
    Dc = np.zeros((Cc.shape[0], 1))

    Q = np.eye(Ac.shape[0])
    R = np.eye(Bc.shape[1] if len(Bc.shape) > 1 else 1)

    Sc = linalg.solve_continuous_are(Ac, Bc.reshape(-1, 1), Q, R,)
    Kc = linalg.solve(R, Bc.T @ Sc).reshape(1, -1)
    ct_ctr = LTISystem(Kc)

    evals = np.sort(np.abs(
        linalg.eig(Ac, left=False, right=False, check_finite=False)
    ))
    dT = 1/(2*evals[-1])

    Tsim = (8/np.min(evals[~np.isclose(evals[np.nonzero(evals)], 0)])
            if np.sum(np.isclose(evals[np.nonzero(evals)], 0)) > 0
            else 8
            )

    dt_data = signal.cont2discrete((Ac, Bc.reshape(-1, 1), Cc, Dc), dT)
    Ad, Bd, Cd, Dd = dt_data[:-1]
    Sd = linalg.solve_discrete_are(Ad, Bd.reshape(-1, 1), Q, R,)
    Kd = linalg.solve(Bd.T @ Sd @ Bd + R, Bd.T @ Sd @ Ad)

    dt_sys = LTISystem(Ad, Bd, dt=dT)
    dt_sys.initial_condition = ct_sys.initial_condition
    dt_ctr = LTISystem(Kd, dt=dT)

    yield ct_sys, ct_ctr, dt_sys, dt_ctr, ref, Tsim


@pytest.fixture(scope="module")
def simulation_results(control_systems, intname):
    ct_sys, ct_ctr, dt_sys, dt_ctr, ref, Tsim = control_systems

    intopts = block_diagram.DEFAULT_INTEGRATOR_OPTIONS.copy()
    intopts['name'] = intname

    if intname == 'dopri5':
        tspan = Tsim
    elif intname == 'lsoda':
        tspan = np.arange(0, Tsim, dt_sys.dt*2**-2)

    results = []
    for sys, ctr in [(dt_sys, dt_ctr), (ct_sys, ct_ctr), (ct_sys, dt_ctr)]:
        bd = BlockDiagram(sys, ref, ctr)
        bd.connect(sys, ref)
        bd.connect(ref, ctr)
        bd.connect(ctr, sys)
        results.append(bd.simulate(tspan, integrator_options=intopts))

    yield results, ct_sys, ct_ctr, dt_sys, dt_ctr, ref, Tsim, tspan, intname
