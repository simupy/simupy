import pytest
import numpy as np
import numpy.testing as npt
from simupy.systems import DynamicalSystem

state_equation_function = lambda *args: np.ones(np.r_[args].shape)
output_equation_function = lambda *args: np.ones(np.r_[args].shape)


def test_no_output():
    with pytest.raises(ValueError):
        DynamicalSystem(dim_state=0)

def test_no_state_equation_function():
    with pytest.raises(ValueError):
        DynamicalSystem(dim_state=1)
    DynamicalSystem(dim_state=1,
                    state_equation_function=state_equation_function)

def test_no_output_equation_function():
    N = 4
    sys = DynamicalSystem(dim_state=1,
                          state_equation_function=state_equation_function)
    args = np.random.rand(N)
    npt.assert_allclose(sys.output_equation_function(args), args[1:])
    sys = DynamicalSystem(dim_state=1,
                          state_equation_function=state_equation_function,
                          output_equation_function=output_equation_function)
    npt.assert_allclose(sys.output_equation_function(args), np.ones(N))

