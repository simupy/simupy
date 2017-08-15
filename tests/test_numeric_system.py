import pytest
import numpy as np
import numpy.testing as npt
from simupy.systems import (DynamicalSystem, need_state_equation_function_msg,
                            need_output_equation_function_msg,
                            zero_dim_output_msg)


N = 4
state_equation_function = lambda t, x, u: np.ones(x.shape)
output_equation_function = lambda t, u: np.ones(u.shape)


def test_dim_output_0():
    with pytest.raises(ValueError, match=zero_dim_output_msg):
        DynamicalSystem(dim_state=0,
                        output_equation_function=output_equation_function)

def test_state_equation_function_kwarg():
    with pytest.raises(ValueError, match=need_state_equation_function_msg):
        DynamicalSystem(dim_state=1)
    DynamicalSystem(dim_state=1,
                    state_equation_function=state_equation_function)

def test_output_equation_function_kwarg():
    with pytest.raises(ValueError, match=need_output_equation_function_msg):
        DynamicalSystem(dim_output=1)
    sys = DynamicalSystem(dim_state=1,
                          state_equation_function=state_equation_function)
    args = np.random.rand(N+1)
    npt.assert_allclose(
        sys.output_equation_function(args[0], args[1:]),
        args[1:]
    )
    sys = DynamicalSystem(dim_state=1,
                          state_equation_function=state_equation_function,
                          output_equation_function=output_equation_function)
    npt.assert_allclose(
        sys.output_equation_function(args[0], args[1:]),
        np.ones(N)
    )

def test_initial_condition_kwarg():
    sys = DynamicalSystem(dim_state=N,
                          state_equation_function=state_equation_function)
    npt.assert_allclose(sys.initial_condition, np.zeros(N))
    sys = DynamicalSystem(dim_state=N,
                          initial_condition=np.ones(N),
                          state_equation_function=state_equation_function)
    npt.assert_allclose(sys.initial_condition, np.ones(N))

