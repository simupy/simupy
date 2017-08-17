import pytest
import numpy as np
import numpy.testing as npt
from simupy.systems import (SwitchedSystem, need_state_equation_function_msg,
                            need_output_equation_function_msg,
                            zero_dim_output_msg, full_state_output)

max_n_condition = 4
bounds_min = -1
bounds_max = 1


def state_equation_function(t, x, u):
    return np.ones(x.shape)


def output_equation_function(t, u):
    return np.ones(u.shape)


def event_variable_equation_function(t, x):
    return x


@pytest.fixture(scope="module", params=np.arange(2, max_n_condition))
def switch_fixture(request):
    yield (((bounds_max - bounds_min)*np.sort(np.random.rand(request.param-1))
           + bounds_min),
           np.array([lambda t, x, u: cnd*np.ones(x.shape)
                     for cnd in range(request.param)]),
           np.array([lambda t, u: cnd*np.ones(u.shape)
                     for cnd in range(request.param)])
           )


def test_dim_output_0(switch_fixture):
    with pytest.raises(ValueError, match=zero_dim_output_msg):
        SwitchedSystem(
            event_variable_equation_function=event_variable_equation_function,
            event_bounds=switch_fixture[0],
            output_equations_functions=switch_fixture[2]
        )
    SwitchedSystem(
        dim_output=1,
        event_variable_equation_function=event_variable_equation_function,
        event_bounds=switch_fixture[0],
        output_equations_functions=switch_fixture[2],
    )


def test_state_equations_functions_kwarg(switch_fixture):
    with pytest.raises(ValueError, match=need_state_equation_function_msg):
        SwitchedSystem(
            dim_state=1,
            event_variable_equation_function=event_variable_equation_function,
            event_bounds=switch_fixture[0],
            output_equations_functions=switch_fixture[2]
        )
    with pytest.raises(ValueError, match="broadcast"):
        SwitchedSystem(
            dim_state=1,
            event_variable_equation_function=event_variable_equation_function,
            event_bounds=switch_fixture[0],
            state_equations_functions=np.array([
                     lambda t, x, u: cnd*np.ones(x.shape)
                     for cnd in range(switch_fixture[0].size+2)
             ]),
            output_equations_functions=switch_fixture[2]
        )
    sys = SwitchedSystem(
        dim_state=1,
        event_variable_equation_function=event_variable_equation_function,
        event_bounds=switch_fixture[0],
        state_equations_functions=switch_fixture[1],
        output_equations_functions=switch_fixture[2]
    )
    npt.assert_array_equal(sys.state_equations_functions, switch_fixture[1])
    sys = SwitchedSystem(
        dim_state=1,
        event_variable_equation_function=event_variable_equation_function,
        event_bounds=switch_fixture[0],
        state_equations_functions=state_equation_function,
        output_equations_functions=switch_fixture[2]
    )
    npt.assert_array_equal(
        sys.state_equations_functions,
        state_equation_function
    )


def test_output_equations_functions_kwarg(switch_fixture):
    with pytest.raises(ValueError, match=need_output_equation_function_msg):
        SwitchedSystem(
            dim_output=1,
            event_variable_equation_function=event_variable_equation_function,
            event_bounds=switch_fixture[0],
        )
    with pytest.raises(ValueError, match="broadcast"):
        SwitchedSystem(
            dim_output=1,
            event_variable_equation_function=event_variable_equation_function,
            event_bounds=switch_fixture[0],
            output_equations_functions=np.array([
                     lambda t, u: cnd*np.ones(u.shape)
                     for cnd in range(switch_fixture[0].size+2)
             ]),
        )
    sys = SwitchedSystem(
        dim_state=1,
        event_variable_equation_function=event_variable_equation_function,
        event_bounds=switch_fixture[0],
        state_equations_functions=switch_fixture[1],
    )
    npt.assert_array_equal(
        sys.output_equations_functions,
        full_state_output
    )

    sys = SwitchedSystem(
        dim_output=1,
        event_variable_equation_function=event_variable_equation_function,
        event_bounds=switch_fixture[0],
        output_equations_functions=switch_fixture[2]
    )
    npt.assert_array_equal(sys.output_equations_functions, switch_fixture[2])

    sys = SwitchedSystem(
        dim_output=1,
        event_variable_equation_function=event_variable_equation_function,
        event_bounds=switch_fixture[0],
        output_equations_functions=output_equation_function
    )
    npt.assert_array_equal(
        sys.output_equations_functions,
        output_equation_function
    )


def test_event_equation_function(switch_fixture):
    sys = SwitchedSystem(
        dim_output=1,
        event_variable_equation_function=event_variable_equation_function,
        event_bounds=switch_fixture[0],
        state_equations_functions=switch_fixture[1],
        output_equations_functions=switch_fixture[2],
    )

    assert sys.state_update_equation_function == full_state_output

    for x in np.linspace(bounds_min, bounds_max):
        if not np.any(np.isclose(x, switch_fixture[0])):
            assert ~np.any(np.isclose(
                sys.event_equation_function(x, x),
                0
            ))

    for zero in switch_fixture[0]:
        npt.assert_allclose(
            sys.event_equation_function(len(switch_fixture[0]), zero),
            0
        )


def test_update_equation_function(switch_fixture):
    sys = SwitchedSystem(
        dim_output=1,
        event_variable_equation_function=event_variable_equation_function,
        event_bounds=switch_fixture[0],
        state_equations_functions=switch_fixture[1],
        output_equations_functions=switch_fixture[2],
    )

    assert not hasattr(sys, 'condition_idx')
    sys.prepare_to_integrate()

    assert sys.condition_idx is None
    sys.update_equation_function(np.random.rand(1), bounds_min)
    assert sys.condition_idx == 0

    for cnd_idx, zero in enumerate(switch_fixture[0]):
        sys.update_equation_function(np.random.rand(1), zero)
        assert sys.condition_idx == cnd_idx+1

    if len(switch_fixture[0]) > 1:
        with pytest.warns(UserWarning):
            sys.update_equation_function(np.random.rand(1), bounds_min)
