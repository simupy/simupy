import sympy as sp
import numpy as np
from simupy.systems.symbolic import (DynamicalSystem, MemorylessSystem,
                                     dynamicsymbols, find_dynamicsymbols)
from simupy.systems import SwitchedSystem as SwitchedSystemBase


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
                 state_equations=None, output_equations=None, **kwargs):
        """
        SwitchedSystem constructor, used to create switched systems from
        symbolic expressions.

        Parameters
        ----------
        event_variable_equation : sympy Expression
            Expression representing the event_equation_function
        event_bounds_expressions : list-like of sympy Expressions or floats
            Ordered list-like values which define the boundaries of events
            (relative to event_variable_equation).
        output_equations : Array or Matrix (2D) of sympy Expressions
            The output equations of the system. The first dimension indexes the
            event-state and should be one more than the number of event bounds.
            This should also be indexed to match the boundaries (i.e., the
            first expression is used when the event_variable_equation is below
            the first event_bounds value). The second dimension is dim_output
            of the system.
        input_ : Array or Matrix (1D) of sympy symbols
            The input of the systems. event_variable_equation and
            output_equations depend on the system's input.
        """
        DiscontinuousSystem.__init__(self, **kwargs)
        self.event_variable_equation = event_variable_equation
        self.output_equations = output_equations
        self.state_equations = state_equations
        self.event_bounds_expressions = event_bounds_expressions
        self.condition_idx = None

    @property
    def state_equations(self):
        return self._state_equations

    @state_equations.setter
    def state_equations(self, state_equations):
        if state_equations is None:  # or other checks?
            self._state_equations = None
            return
        if hasattr(self, 'event_bounds'):
            n_conditions_test = self.event_bounds.shape[0]+1
            assert state_equations.shape[0] == n_conditions_test
        self._state_equations = state_equations
        self.n_conditions = state_equations.shape[0]
        self.state_equations_functions = np.empty(self.n_conditions, object)

        self._state_equation_function = self.state_equation_function
        for cond_idx in range(self.n_conditions):
            self.state_equation = state_equations[cond_idx]
            self.state_equations_functions[cond_idx] = \
                self.output_equation_function
        self.state_equation_function = self._state_equation_function

    @property
    def output_equations(self):
        return self._output_equations

    @output_equations.setter
    def output_equations(self, output_equations):
        if output_equations is None:  # or other checks?
            self._output_equations = None
            return
        if hasattr(self, 'event_bounds'):
            n_conditions_test = self.event_bounds.shape[0]+1
            assert output_equations.shape[0] == n_conditions_test
        self._output_equations = output_equations
        self.n_conditions = output_equations.shape[0]
        self.output_equations_functions = np.empty(self.n_conditions, object)

        self._output_equation_function = self.output_equation_function
        for cond_idx in range(self.n_conditions):
            self.output_equation = output_equations[cond_idx]
            self.output_equations_functions[cond_idx] = \
                self.output_equation_function
        self.output_equation_function = self._output_equation_function

    @property
    def event_variable_equation(self):
        return self._event_variable_equation

    @event_variable_equation.setter
    def event_variable_equation(self, event_variable_equation):
        if self.dim_state:
            assert find_dynamicsymbols(event_variable_equation) <= \
                set(self.state)
        else:
            assert find_dynamicsymbols(event_variable_equation) <= \
                set(self.input)
        assert event_variable_equation.atoms(sp.Symbol) <= set(
            self.constants_values.keys()) | set([dynamicsymbols._t])
        self._event_variable_equation = event_variable_equation
        self.event_variable_equation_function = self.code_generator(
                [dynamicsymbols._t] + sp.flatten(self.input),
                self._event_variable_equation.subs(self.constants_values),
                **self.code_generator_args
        )

    @property
    def event_bounds_expressions(self):
        return self._event_bounds_expressions

    @event_bounds_expressions.setter
    def event_bounds_expressions(self, event_bounds):
        if hasattr(self, 'output_equations'):
            assert len(event_bounds)+1 == self.output_equations.shape[0]
        if hasattr(self, 'output_equations_functions'):
            assert len(event_bounds)+1 == self.output_equations_functions.size
        self.event_bounds = np.array(
            [sp.N(bound, subs=self.constants_values)
             for bound in event_bounds],
            dtype=np.float_
        )


class MemorylessDiscontinuousSystem(DiscontinuousSystem, MemorylessSystem):
    pass


class SwitchedOutput(SwitchedSystem, MemorylessDiscontinuousSystem):
    """
    A memoryless discontinuous system to conveninetly construct switched
    outputs.
    """


class Saturation(SwitchedOutput):
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Deadband(SwitchedOutput):
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Hysteresis(SwitchedOutput):
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Stiction(SwitchedOutput):
    """
    inputs: v - velocity of object, fi - sum of other forces
    output: fo - force experienced by object (what changes momentum)
    assume everything is pointing in positive direction

    parameters: ffr - basic friction force, mu*Fn. phi - break away force
    factor (positive, constant, real numbers. although could easily change it
    to allow ffr to be a signal too, to model changing mass or whatever)

    v > 0:
        fo = fi - ffr
    v < 0:
        fo = fi + ffr
    v == 0:
        fi < -phi*ffr:
            fo = fi + phi*ffr
        phi*ffr > fi > -phi*ffr:
            fo = 0
        fi > phi*ffr:
            fo = fi - phi*ffr
    """
