import sympy as sp
import numpy as np
from simupy.systems.symbolic import (DynamicalSystem, MemorylessSystem,
                                     dynamicsymbols, find_dynamicsymbols)
from simupy.systems import SwitchedSystem as SwitchedSystemBase
from simupy.array import r_


class DiscontinuousSystem(DynamicalSystem):
    """
    A continuous-time dynamical system with a discontinuity. Must provide the
    following attributes in addition to those of DynamicalSystem:

    ``event_equation_function`` - A function called at each integration time-
    step and stored in simulation results. Takes input and state, if stateful.
    A zero-crossing of this output triggers the discontinuity.

    ``event_equation_function`` - A function that is called when the
    discontinuity occurs. This is generally used to change what
    ``state_equation_function``, ``output_equation_function``, and
    ``event_equation_function`` compute based on the occurance of the
    discontinuity. If stateful, returns the state immediately after the
    discontinuity.
    """

    def event_equation_function(self, *args, **kwargs):
        raise NotImplementedError

    def update_equation_function(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def dt(self):
        return 0

    @dt.setter
    def dt(self, dt):
        if dt != 0:
            raise ValueError("Discontinuous systems only make sense for " +
                             "continuous time systems")


class SwitchedSystem(SwitchedSystemBase, DiscontinuousSystem):
    def __init__(self, event_variable_equation, event_bounds_expressions,
                 state_equations=None, output_equations=None,
                 state_update_equation=None, **kwargs):
        """
        SwitchedSystem constructor, used to create switched systems from
        symbolic expressions. The parameters below are in addition to
        parameters from the ``systems.symbolic.DynamicalSystems`` constructor.

        Parameters
        ----------
        event_variable_equation : sympy Expression
            Expression representing the event_equation_function
        event_bounds_expressions : list-like of sympy Expressions or floats
            Ordered list-like values which define the boundaries of events
            (relative to event_variable_equation).
        state_equations : array_like of sympy Expressions, optional
            The state equations of the system. The first dimension indexes the
            event-state and should be one more than the number of event bounds.
            This should also be indexed to match the boundaries (i.e., the
            first expression is used when the event_variable_equation is below
            the first event_bounds value). The second dimension is dim_state
            of the system. If only 1-D, uses single equation for every
            condition.
        output_equations : array_like of sympy Expressions, optional
            The output equations of the system. The first dimension indexes the
            event-state and should be one more than the number of event bounds.
            This should also be indexed to match the boundaries (i.e., the
            first expression is used when the event_variable_equation is below
            the first event_bounds value). The second dimension is dim_output
            of the system. If only 1-D, uses single equation for every
            condition.
        state_update_equation : sympy Expression
            Expression representing the state_update_equation_function
        """
        DiscontinuousSystem.__init__(self, **kwargs)
        self.event_variable_equation = event_variable_equation
        self.event_bounds_expressions = event_bounds_expressions
        self.output_equations = output_equations
        self.state_equations = state_equations
        self.state_update_equation = state_update_equation
        self.condition_idx = None
        self.validate(True)

    def prepare_to_integrate(self):
        # TODO: refactor the setters so I can call an update instead
        self.event_variable_equation = self.event_variable_equation
        self.event_bounds_expressions = self.event_bounds_expressions
        self.output_equations = self.output_equations
        self.state_equations = self.state_equations
        self.state_update_equation = self.state_update_equation

        super().prepare_to_integrate()


    def validate(self, from_self=False):
        if from_self:
            super().validate()

    @property
    def state_equations(self):
        return self._state_equations

    @state_equations.setter
    def state_equations(self, state_equations):
        if state_equations is None:  # or other checks?
            self._state_equations = None
            return
        if hasattr(self, 'event_bounds'):
            if (len(state_equations.shape) == 1 or
                    state_equations.shape[0] == 1):
                state_equations = r_.__getitem__(
                    ('0,2', *tuple(self.n_conditions*(state_equations,)))
                )
            n_conditions_test = self.event_bounds.shape[1]+1
            assert state_equations.shape[0] == n_conditions_test
        self._state_equations = state_equations
        self.n_conditions = state_equations.shape[0]
        self.state_equations_functions = np.empty(self.n_conditions, object)

        for cond_idx in range(self.n_conditions):
            self.state_equation = state_equations[cond_idx, :]
            self.state_equations_functions[cond_idx] = \
                self.state_equation_function
        self.state_equation_function = (
            lambda *args:
            SwitchedSystemBase.state_equation_function(self, *args)
        )

    @property
    def output_equations(self):
        return self._output_equations

    @output_equations.setter
    def output_equations(self, output_equations):
        if output_equations is None:  # or other checks?
            if self.dim_state > 0 and hasattr(self, 'n_conditions'):
                output_equations = r_.__getitem__(
                    ('0,2', *tuple(self.n_conditions*(self.state,)))
                )
            else:
                self._output_equations = None
                return
        if hasattr(self, 'event_bounds'):
            if (len(output_equations.shape) == 1 or
                    output_equations.shape[0] == 1):
                output_equations = r_.__getitem__(
                    ('0,2', *tuple(self.n_conditions*(output_equations,)))
                )
            n_conditions_test = self.event_bounds.shape[1]+1
            assert output_equations.shape[0] == n_conditions_test
        self._output_equations = output_equations
        self.n_conditions = output_equations.shape[0]
        self.output_equations_functions = np.empty(self.n_conditions, object)

        for cond_idx in range(self.n_conditions):
            self.output_equation = output_equations[cond_idx, :]
            self.output_equations_functions[cond_idx] = \
                self.output_equation_function
        self.output_equation_function = (
            lambda *args:
            SwitchedSystemBase.output_equation_function(self, *args)
        )

    @property
    def state_update_equation(self):
        return self._state_update_equation

    @state_update_equation.setter
    def state_update_equation(self, state_update_equation):
        if state_update_equation is None:
            if self.dim_state > 0:
                state_update_equation = self.state
            else:
                state_update_equation = self.input
        assert state_update_equation.atoms(sp.Symbol) <= set(
            self.constants_values.keys()) | set([dynamicsymbols._t])
        self._state_update_equation = state_update_equation
        if self.dim_state:
            assert find_dynamicsymbols(state_update_equation) <= \
                set(self.state) | set(self.input)
            self.state_update_equation_function = self.code_generator(
                    ([dynamicsymbols._t] + 
                     sp.flatten(self.state) + sp.flatten(self.input)),
                    self._state_update_equation.subs(self.constants_values),
                    **self.code_generator_args
            )
        else:
            assert find_dynamicsymbols(state_update_equation) <= \
                set(self.input)
            self.state_update_equation_function = self.code_generator(
                    [dynamicsymbols._t] + sp.flatten(self.input),
                    self._state_update_equation.subs(self.constants_values),
                    **self.code_generator_args
            )

    @property
    def event_variable_equation(self):
        return self._event_variable_equation

    @event_variable_equation.setter
    def event_variable_equation(self, event_variable_equation):
        assert event_variable_equation.atoms(sp.Symbol) <= set(
            self.constants_values.keys()) | set([dynamicsymbols._t])
        self._event_variable_equation = event_variable_equation
        if self.dim_state:
            assert find_dynamicsymbols(event_variable_equation) <= \
                set(self.state)
            self.event_variable_equation_function = self.code_generator(
                [dynamicsymbols._t] + sp.flatten(self.state),
                self._event_variable_equation.subs(self.constants_values),
                **self.code_generator_args
            )
        else:
            assert find_dynamicsymbols(event_variable_equation) <= \
                set(self.input)
            self.event_variable_equation_function = self.code_generator(
                [dynamicsymbols._t] + sp.flatten(self.input),
                self._event_variable_equation.subs(self.constants_values),
                **self.code_generator_args
            )

    @property
    def event_bounds_expressions(self):
        return self._event_bounds_expressions

    @event_bounds_expressions.setter
    def event_bounds_expressions(self, event_bounds_exp):
        if hasattr(self, 'output_equations'):
            assert len(event_bounds_exp)+1 == self.output_equations.shape[0]
        if hasattr(self, 'output_equations_functions'):
            assert len(event_bounds_exp)+1 == \
                self.output_equations_functions.size
        if getattr(self, 'state_equations', None) is not None:
            assert len(event_bounds_exp)+1 == self.state_equations.shape[0]
        if getattr(self, 'state_equations_functions', None) is not None:
            assert len(event_bounds_exp)+1 == \
                self.state_equations_functions.size
        self._event_bounds_expressions = event_bounds_exp
        self.event_bounds = np.array(
            [sp.N(bound, subs=self.constants_values)
             for bound in event_bounds_exp],
            dtype=np.float_
        )


class MemorylessDiscontinuousSystem(DiscontinuousSystem, MemorylessSystem):
    pass


class SwitchedOutput(SwitchedSystem, MemorylessDiscontinuousSystem):
    """
    A memoryless discontinuous system to conveninetly construct switched
    outputs.
    """
