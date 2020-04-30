import pytest
import numpy as np
import sympy as sp
import numpy.testing as npt
from scipy import linalg, signal
from simupy.systems import LTISystem, SystemFromCallable
from simupy.systems.symbolic import dynamicsymbols
from simupy.discontinuities import SwitchedSystem
from simupy.array import Array, r_
from simupy.utils import (callable_from_trajectory,
                          array_callable_from_vector_trajectory)
from simupy.matrices import construct_explicit_matrix, system_from_matrix_DE
import simupy.block_diagram as block_diagram

BlockDiagram = block_diagram.BlockDiagram
block_diagram.DEFAULT_INTEGRATOR_OPTIONS['rtol'] = 1E-9

TEST_ATOL = 1E-6
TEST_RTOL = 1E-6

RICCATI_TEST_RTOL = 2E-2
RICCATI_TEST_ATOL = 0


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

        # get_double_integrator(m=1, d=1, b=4.0),  # overdamped
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

    Tsim = (8/np.min(evals[~np.isclose(evals, 0)])
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


@pytest.fixture(scope="module", params=['dopri5', 'lsoda'])
def intname(request):
    if request.param == 'lsoda':
        pytest.xfail("Only support adaptive step-size and dense output.")
    yield request.param


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

@pytest.mark.xfail
def test_fixed_integration_step_equivalent(control_systems):
    """
    using the a list-like tspan or float like tspan should give the same
    results
    """
    ct_sys, ct_ctr, dt_sys, dt_ctr, ref, Tsim = control_systems

    for sys, ctr in [(ct_sys, ct_ctr), (ct_sys, dt_ctr)]:
        bd = BlockDiagram(sys, ref, ctr)
        bd.connect(sys, ref)
        bd.connect(ref, ctr)
        bd.connect(ctr, sys)

        var_res = bd.simulate(Tsim)
        fix_res = bd.simulate(np.arange(1+Tsim//dt_sys.dt)*dt_sys.dt)

        if ctr == ct_ctr:
            unique_t, unique_t_sel = np.unique(
                var_res.t, return_index=True
            )
            ct_res = callable_from_trajectory(
                unique_t, var_res.x[unique_t_sel, :]
            )

            npt.assert_allclose(
                ct_res(fix_res.t), fix_res.x,
                atol=TEST_ATOL, rtol=TEST_RTOL
            )
        else:
            var_sel = np.where(
                np.equal(*np.meshgrid(var_res.t, fix_res.t))
            )[1][::2]

            npt.assert_allclose(
                var_res.x[var_sel, :], fix_res.x,
                atol=TEST_ATOL, rtol=TEST_RTOL
            )

    bd = BlockDiagram(dt_sys, ref, dt_ctr)
    bd.connect(dt_sys, ref)
    bd.connect(ref, dt_ctr)
    bd.connect(dt_ctr, dt_sys)

    var_res = bd.simulate(Tsim)
    fix_res = bd.simulate(var_res.t[var_res.t < Tsim])

    npt.assert_allclose(
        var_res.x[var_res.t < Tsim], fix_res.x,
        atol=TEST_ATOL, rtol=TEST_RTOL
    )

##
def test_feedback_equivalent(simulation_results):
    # (A-BK) should be exactly same as system A,B under feedback K
    results, ct_sys, ct_ctr, dt_sys, dt_ctr, ref, Tsim, tspan, intname = \
        simulation_results

    intopts = block_diagram.DEFAULT_INTEGRATOR_OPTIONS.copy()
    intopts['name'] = intname

    dt_equiv_sys = LTISystem(dt_sys.F + dt_sys.G @ dt_ctr.K,
                             -dt_sys.G @ dt_ctr.K, dt=dt_sys.dt)
    dt_equiv_sys.initial_condition = dt_sys.initial_condition

    dt_bd = BlockDiagram(dt_equiv_sys, ref)
    dt_bd.connect(ref, dt_equiv_sys)
    dt_equiv_res = dt_bd.simulate(tspan, integrator_options=intopts)

    mixed_t_discrete_t_equal_idx = np.where(
        np.equal(*np.meshgrid(dt_equiv_res.t, results[0].t))
    )[1]


    npt.assert_allclose(
        dt_equiv_res.x, results[0].x,
        atol=TEST_ATOL, rtol=TEST_RTOL
    )

    ct_equiv_sys = LTISystem(ct_sys.F + ct_sys.G @ ct_ctr.K,
                             -ct_sys.G @ ct_ctr.K)
    ct_equiv_sys.initial_condition = ct_sys.initial_condition

    ct_bd = BlockDiagram(ct_equiv_sys, ref)
    ct_bd.connect(ref, ct_equiv_sys)
    ct_equiv_res = ct_bd.simulate(tspan, integrator_options=intopts)
    unique_t, unique_t_sel = np.unique(ct_equiv_res.t, return_index=True)
    ct_res = callable_from_trajectory(
        unique_t,
        ct_equiv_res.x[unique_t_sel, :]
    )

    npt.assert_allclose(
        ct_res(results[1].t), results[1].x,
        atol=TEST_ATOL, rtol=TEST_RTOL
    )


def test_dt_ct_equivalent(simulation_results):
    """
    CT system should match DT equivalent exactly at t=k*dT under the same
    inputs.
    """
    results, ct_sys, ct_ctr, dt_sys, dt_ctr, ref, Tsim, tspan, intname = \
        simulation_results

    dt_unique_t, dt_unique_t_idx = np.unique(
        results[0].t, return_index=True
    )
    discrete_sel = dt_unique_t_idx[
        (dt_unique_t < (Tsim*7/8)) & (dt_unique_t % dt_sys.dt == 0)
    ]

    mixed_t_discrete_t_equal_idx = np.where(
        np.equal(*np.meshgrid(results[2].t, results[0].t[discrete_sel]))
    )[1]

    mixed_unique_t, mixed_unique_t_idx_rev = np.unique(
        results[2].t[mixed_t_discrete_t_equal_idx][::-1], return_index=True
    )
    mixed_unique_t_idx = (len(mixed_t_discrete_t_equal_idx)
                          - mixed_unique_t_idx_rev - 1)
    mixed_sel = mixed_t_discrete_t_equal_idx[mixed_unique_t_idx]

    npt.assert_allclose(
        results[2].t[mixed_sel], results[0].t[discrete_sel],
        atol=TEST_ATOL, rtol=TEST_RTOL
    )

    npt.assert_allclose(
        results[2].x[mixed_sel, :], results[0].x[discrete_sel, :],
        atol=TEST_ATOL, rtol=TEST_RTOL
    )


def test_mixed_dts(simulation_results):
    """
    A DT equivalent that is sampled twice as fast should match original DT
    equivalent exactly at t= k*dT under the same inputs.
    """
    results, ct_sys, ct_ctr, dt_sys, dt_ctr, ref, Tsim, tspan, intname = \
        simulation_results
    Ac, Bc, Cc = ct_sys.data
    Dc = np.zeros((Cc.shape[0], 1))

    dt_unique_t, dt_unique_t_idx = np.unique(
        results[0].t, return_index=True
    )
    discrete_sel = dt_unique_t_idx[
        (dt_unique_t < (Tsim*7/8)) & (dt_unique_t % dt_sys.dt == 0)
    ]

    scale_factor = 1/2
    Ad, Bd, Cd, Dd, dT = signal.cont2discrete(
        (Ac, Bc.reshape(-1, 1), Cc, Dc),
        dt_sys.dt*scale_factor
    )
    dt_sys2 = LTISystem(Ad, Bd, dt=dT)
    dt_sys2.initial_condition = ct_sys.initial_condition

    intopts = block_diagram.DEFAULT_INTEGRATOR_OPTIONS.copy()
    intopts['name'] = intname

    bd = BlockDiagram(dt_sys2, ref, dt_ctr)
    bd.connect(dt_sys2, ref)
    bd.connect(ref, dt_ctr)
    bd.connect(dt_ctr, dt_sys2)
    res = bd.simulate(tspan, integrator_options=intopts)

    mixed_t_discrete_t_equal_idx = np.where(
        np.equal(*np.meshgrid(res.t, results[0].t[discrete_sel]))
    )[1]

    mixed_unique_t, mixed_unique_t_idx_rev = np.unique(
        res.t[mixed_t_discrete_t_equal_idx][::-1], return_index=True
    )
    mixed_unique_t_idx = (len(mixed_t_discrete_t_equal_idx)
                          - mixed_unique_t_idx_rev - 1)
    mixed_sel = mixed_t_discrete_t_equal_idx[mixed_unique_t_idx]

    npt.assert_allclose(
        res.x[mixed_sel, :], results[0].x[discrete_sel, :],
        atol=TEST_ATOL, rtol=TEST_RTOL
    )


def test_riccati_algebraic_equation(control_systems):
    ct_sys, ct_ctr, dt_sys, dt_ctr, ref, Tsim = control_systems
    n = ct_sys.dim_state
    m = ct_sys.dim_input

    S = construct_explicit_matrix('s', n, n, symmetric=True, dynamic=True)
    Q = np.eye(n)
    R = np.eye(m)
    Rinv = np.linalg.inv(R)
    A, B, C = ct_sys.data

    Sdot = (Q+A.T*S+S*A-S*B*Rinv*B.T*S)
    ct_riccati_sys = system_from_matrix_DE(Sdot, S)
    ct_riccati_bd = BlockDiagram(ct_riccati_sys)
    ct_riccati_res = ct_riccati_bd.simulate(2*Tsim)

    ct_sim_data_unique_t, ct_sim_data_unique_t_idx = np.unique(
        ct_riccati_res.t,
        return_index=True
    )

    mat_ct_s_callable = array_callable_from_vector_trajectory(
        np.flipud(Tsim-ct_riccati_res.t[ct_sim_data_unique_t_idx]),
        np.flipud(ct_riccati_res.x[ct_sim_data_unique_t_idx]),
        ct_riccati_sys.state,
        S
    )

    K = Rinv @ B.T @ mat_ct_s_callable(0)
    npt.assert_allclose(
        K, ct_ctr.K, atol=RICCATI_TEST_ATOL, rtol=RICCATI_TEST_RTOL
    )


def test_events():
    # use bouncing ball to test events work

    # simulate in block diagram
    int_opts = block_diagram.DEFAULT_INTEGRATOR_OPTIONS.copy()
    int_opts['rtol'] = 1E-12
    int_opts['atol'] = 1E-15
    int_opts['nsteps'] = 1000
    int_opts['max_step'] = 2**-3
    x = x1, x2 = Array(dynamicsymbols('x_1:3'))
    mu, g = sp.symbols('mu g')
    constants = {mu: 0.8, g: 9.81}
    ic = np.r_[10, 15]
    sys = SwitchedSystem(
        x1, Array([0]),
        state_equations=r_[x2, -g],
        state_update_equation=r_[sp.Abs(x1), -mu*x2],
        state=x,
        constants_values=constants,
        initial_condition=ic
    )
    bd = BlockDiagram(sys)
    res = bd.simulate(5, integrator_options=int_opts)

    # compute actual impact time
    tvar = dynamicsymbols._t
    impact_eq = (x2*tvar - g*tvar**2/2 + x1).subs(
        {x1: ic[0], x2: ic[1], g: 9.81}
    )
    t_impact = sp.solve(impact_eq, tvar)[-1]

    # make sure simulation actually changes velocity sign around impact
    abs_diff_impact = np.abs(res.t - t_impact)
    impact_idx = np.where(abs_diff_impact == np.min(abs_diff_impact))[0]
    assert np.sign(res.x[impact_idx-1, 1]) != np.sign(res.x[impact_idx+1, 1])
