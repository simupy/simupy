import pytest
import numpy as np
import sympy as sp
from scipy import signal
import numpy.testing as npt
from simupy.systems import LTISystem
from simupy.systems.symbolic import dynamicsymbols
from simupy.discontinuities import SwitchedSystem
from simupy.array import Array, r_
from simupy.utils import callable_from_trajectory
import simupy.block_diagram as block_diagram

from fixture import simulation_results, control_systems

BlockDiagram = block_diagram.BlockDiagram
block_diagram.DEFAULT_INTEGRATOR_OPTIONS['rtol'] = 1E-9

TEST_TOL = 1E-6


@pytest.fixture(scope="module", params=['dopri5', 'lsoda'])
def intname(request):
    yield request.param


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
                atol=TEST_TOL
            )
        else:
            var_sel = np.where(
                np.equal(*np.meshgrid(var_res.t, fix_res.t))
            )[1][::2]

            npt.assert_allclose(
                var_res.x[var_sel, :], fix_res.x,
                atol=TEST_TOL
            )

    bd = BlockDiagram(dt_sys, ref, dt_ctr)
    bd.connect(dt_sys, ref)
    bd.connect(ref, dt_ctr)
    bd.connect(dt_ctr, dt_sys)

    var_res = bd.simulate(Tsim)
    fix_res = bd.simulate(var_res.t[var_res.t < Tsim])

    npt.assert_allclose(
        var_res.x[var_res.t < Tsim], fix_res.x,
        atol=TEST_TOL
    )


def test_feedback_equivalent(simulation_results):
    # (A-BK) should be exactly same as system A,B under feedback K
    results, ct_sys, ct_ctr, dt_sys, dt_ctr, ref, Tsim, tspan, intname = \
        simulation_results

    intopts = block_diagram.DEFAULT_INTEGRATOR_OPTIONS.copy()
    intopts['name'] = intname

    dt_equiv_sys = LTISystem(dt_sys.F - dt_sys.G @ dt_ctr.K,
                             dt_sys.G @ dt_ctr.K, dt=dt_sys.dt)
    dt_equiv_sys.initial_condition = dt_sys.initial_condition

    dt_bd = BlockDiagram(dt_equiv_sys, ref)
    dt_bd.connect(ref, dt_equiv_sys)
    dt_equiv_res = dt_bd.simulate(tspan, integrator_options=intopts)
    npt.assert_allclose(
        dt_equiv_res.x, results[0].x
    )

    ct_equiv_sys = LTISystem(ct_sys.F - ct_sys.G @ ct_ctr.K,
                             ct_sys.G @ ct_ctr.K)
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
        atol=TEST_TOL
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
        results[2].x[mixed_sel, :], results[0].x[discrete_sel, :],
        atol=TEST_TOL
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
        atol=TEST_TOL
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
