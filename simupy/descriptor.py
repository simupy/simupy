import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics.functions import find_dynamicsymbols
from simupy.systems import DynamicalSystem


class DescriptorSystem(DynamicalSystem):
    """
    A dynamical system in descriptor form, with a mass matrix. Currently this
    is only useful for symbolic analysis. If DAE solvers are supported, the
    mass matrix form could be used directly in numerical integration.
    
    Uses a generalized momentum nomenclature. The system represents dynamics of
    the form::

        M(t,x) * x_dot(t) = f(t,x,u)
        y(t) = h(t,x)

    M is the mass matrix and f is the impulse equations. The state equation is
    automatically solved for so that the DescriptorSystem can be used in place
    of a DynamicalSystem.
    """
    def __init__(self, mass_matrix=None, impulse_equation=None, state=None,
                 input_=None, output_equation=None, **kwargs):

        super().__init__(
            state=state,
            input_=input_,
            output_equation=output_equation,
            **kwargs
        )

        self.impulse_equation = impulse_equation
        self.mass_matrix = mass_matrix

    @property
    def impulse_equation(self):
        return self._impulse_equation

    @impulse_equation.setter
    def impulse_equation(self, impulse_equation):
        assert find_dynamicsymbols(impulse_equation) <= (
                set(self.state) | set(self.input)
            )
        assert impulse_equation.atoms(sp.Symbol) <= (
                set(self.constants_values.keys()) | set([dynamicsymbols._t])
            )
        self._impulse_equation = impulse_equation

    @property
    def mass_matrix(self):
        return self._mass_matrix

    @mass_matrix.setter
    def mass_matrix(self, mass_matrix):
        if mass_matrix is None:
            mass_matrix = sp.eye(self.dim_state)
        assert mass_matrix.shape[1] == len(self.state)
        assert mass_matrix.shape[0] == len(self.impulse_equation)
        assert find_dynamicsymbols(mass_matrix) <= (
                set(self.state) | set(self.input)
            )
        assert mass_matrix.atoms(sp.Symbol) <= (
                set(self.constants_values.keys()) | set([dynamicsymbols._t])
            )

        self.state_equation = mass_matrix.LUsolve(self.impulse_equation)
        self._mass_matrix = mass_matrix
        # TODO: callable for mass matrices and impulse_equation for DAE solvers
