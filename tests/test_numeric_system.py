import pytest
import numpy as np
import numpy.testing as npt
from simupy.systems import (DynamicalSystem, need_state_equation_function_msg,
                            need_output_equation_function_msg,
                            zero_dim_output_msg)


N = 4


def ones_equation_function(t, x):
    return np.ones(x.shape)


def test_dim_output_0():
    with pytest.raises(ValueError, match=zero_dim_output_msg):
        DynamicalSystem(dim_state=0)


def test_state_equation_function_kwarg():
    with pytest.raises(ValueError, match=need_state_equation_function_msg):
        DynamicalSystem(dim_state=N)
    args = np.random.rand(N+1)
    sys = DynamicalSystem(dim_state=N,
                          state_equation_function=ones_equation_function)
    npt.assert_allclose(
        sys.state_equation_function(args[0], args[1:]),
        np.ones(N)
    )


def test_output_equation_function_kwarg():
    with pytest.raises(ValueError, match=need_output_equation_function_msg):
        DynamicalSystem(dim_output=N)

    args = np.random.rand(N+1)

    sys = DynamicalSystem(dim_state=N,
                          state_equation_function=ones_equation_function)
    npt.assert_allclose(
        sys.output_equation_function(args[0], args[1:]),
        args[1:]
    )

    sys = DynamicalSystem(dim_state=1,
                          state_equation_function=ones_equation_function,
                          output_equation_function=ones_equation_function)
    npt.assert_allclose(
        sys.output_equation_function(args[0], args[1:]),
        np.ones(N)
    )


def test_initial_condition_kwarg():
    sys = DynamicalSystem(dim_state=N,
                          state_equation_function=ones_equation_function)
    npt.assert_allclose(sys.initial_condition, np.zeros(N))
    sys = DynamicalSystem(dim_state=N,
                          initial_condition=np.ones(N),
                          state_equation_function=ones_equation_function)
    npt.assert_allclose(sys.initial_condition, np.ones(N))
