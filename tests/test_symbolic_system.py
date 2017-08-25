import pytest
import numpy as np
import sympy as sp
import numpy.testing as npt
from simupy.systems.symbolic import DynamicalSystem, dynamicsymbols
from simupy.systems import (need_state_equation_function_msg,
                            zero_dim_output_msg)
from simupy.array import Array, r_

x = x1, x2 = Array(dynamicsymbols('x1:3'))
mu = sp.symbols('mu')
state_equation = r_[x2, -x1+mu*(1-x1**2)*x2]
output_equation = r_[x1**2 + x2**2, sp.atan2(x2, x1)]
constants = {mu: 5}


def test_dim_output_0():
    with pytest.raises(ValueError, match=zero_dim_output_msg):
        DynamicalSystem(input_=x, constants_values=constants)


def test_state_equation_kwarg():
    with pytest.raises(ValueError, match=need_state_equation_function_msg):
        DynamicalSystem(state=x, constants_values=constants)
    sys = DynamicalSystem(state=x,
                          state_equation=state_equation,
                          constants_values=constants)
    args = np.random.rand(len(x)+1)
    npt.assert_allclose(
        sys.state_equation_function(args[0], args[1:]).squeeze(),
        np.r_[args[2], -args[1]+constants[mu]*(1-args[1]**2)*args[2]]
    )


def test_output_equation_function_kwarg():
    with pytest.raises(ValueError, match=zero_dim_output_msg):
        DynamicalSystem(input_=x)
    args = np.random.rand(len(x)+1)
    sys = DynamicalSystem(state=x,
                          state_equation=state_equation,
                          constants_values=constants)
    npt.assert_allclose(
        sys.output_equation_function(args[0], args[1:]).squeeze(),
        args[1:]
    )
    sys = DynamicalSystem(state=x,
                          state_equation=state_equation,
                          output_equation=output_equation,
                          constants_values=constants)
    npt.assert_allclose(
        sys.output_equation_function(args[0], args[1:]).squeeze(),
        np.r_[args[1]**2 + args[2]**2, np.arctan2(args[2], args[1])]
    )
