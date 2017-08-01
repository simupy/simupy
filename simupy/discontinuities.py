import sympy as sp
import numpy as np
from simupy.systems import (DynamicalSystem, MemorylessSystem, dynamicsymbols,
                            find_dynamicsymbols)


class DiscontinuousSystem(DynamicalSystem):
    """
    now state_equation and output_equation must be n_events x n_state/n_output

    state_equation_function and output_equation_function are always expected
    to return the correct result.

    event_occurance can return a new state if stateful system, otherwise output
    is ignored then updates object (self) properties so that (state,) output,
    event equations

    Is it always enough to know what event-state fit was in and the new event
    crossing? need to know which event changed as well as state at event
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


class MemoryLessDiscontinuousSystem(DiscontinuousSystem, MemorylessSystem):
    pass


class SwitchedOutput(MemoryLessDiscontinuousSystem):
    def __init__(self, event_variable_equation, event_bounds, output_equations,
                 input_=None, *args, **kwargs):
        super().__init__(input_=input_, *args, **kwargs)
        self.event_variable_equation = event_variable_equation
        self.output_equations = output_equations
        self.event_bounds = event_bounds
        self.condition_idx = None

    @property
    def output_equations(self):
        return self._output_equations

    @output_equations.setter
    def output_equations(self, output_equations):
        if hasattr(self, 'event_bounds'):
            n_conditions_test = self.event_bounds.shape[0]+1
            assert output_equations.shape[0] == n_conditions_test
        self._output_equations = output_equations
        self.n_conditions = output_equations.shape[0]
        self.output_equations_functions = np.empty(self.n_conditions, object)
        for cond_idx in range(self.n_conditions):
            self.output_equation = output_equations[cond_idx, :]
            self.output_equations_functions[cond_idx] = \
                self.output_equation_function

        def output_equation_function(t, u):
            return self.output_equations_functions[self.condition_idx](t, u)

        self.output_equation_function = output_equation_function

    @property
    def event_variable_equation(self):
        return self._event_variable_equation

    @event_variable_equation.setter
    def event_variable_equation(self, event_variable_equation):
        assert find_dynamicsymbols(event_variable_equation) <= set(self.input)
        assert event_variable_equation.atoms(sp.Symbol) <= set(
            self.constants_values.keys()) | set([dynamicsymbols._t])
        self._event_variable_equation = event_variable_equation
        self.event_variable_equation_function = self.code_generator(
                [dynamicsymbols._t] + sp.flatten(self.input),
                self._event_variable_equation.subs(self.constants_values),
                **self.code_generator_args
        )

    @property
    def event_bounds(self):
        return self._event_bounds

    @event_bounds.setter
    def event_bounds(self, event_bounds):
        if hasattr(self, 'output_equations'):
            assert len(event_bounds)+1 == self.output_equations.shape[0]
        if hasattr(self, 'output_equations_functions'):
            assert len(event_bounds)+1 == self.output_equations_functions.size
        self._event_bounds = np.array(event_bounds)[None, :]
        self.n_conditions = len(event_bounds) + 1
        if self.n_conditions == 2:
            self.event_bounds_range = self._event_bounds
        else:
            self.event_bounds_range = np.diff(self.event_bounds[0, [0, -1]])

    def event_equation_function(self, t, u):
        event_var = self.event_variable_equation_function(t, u)
        return np.prod(
            (self.event_bounds_range-self.event_bounds)*event_var -
            self.event_bounds*(self.event_bounds_range - event_var),
            axis=1
        )

    def update_equation_function(self, t, u):
        event_var = self.event_variable_equation_function(t, u)
        if self.condition_idx is None:
            self.condition_idx = np.where(np.all(np.r_[
                    np.c_[[[True]], event_var >= self.event_bounds],
                    np.c_[event_var <= self.event_bounds, [[True]]]
                    ], axis=0))[0][0]
        else:
            sq_dist = (event_var - self.event_bounds)**2
            crossed_root_idx = np.where(sq_dist == np.min(sq_dist))[1][0]
            if crossed_root_idx == self.condition_idx:
                self.condition_idx += 1
            elif crossed_root_idx == self.condition_idx -1:
                self.condition_idx -= 1
        return
        
    def prepare_to_integrate(self):
        self.condition_idx = None


class Saturation(MemoryLessDiscontinuousSystem):
    pass


class Deadband(MemoryLessDiscontinuousSystem):
    pass


class Hysteresis(MemoryLessDiscontinuousSystem):
    pass


class Stiction(MemoryLessDiscontinuousSystem):
    """
    inputs: v - velocity of object, fi - sum of other forces
    output: fo - force experienced by object (what changes momentum)
    assume everything is pointing in positive direction

    parameters: ffr - basic friction force, mu*Fn. phi - break away force factor
    (positive, constant, real numbers. although could easily change it to allow
    ffr to be a signal too, to model changing mass or whatever)

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
    pass
