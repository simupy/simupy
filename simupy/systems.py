import sympy as sp
import numpy as np
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics.functions import find_dynamicsymbols
from simupy.utils import process_vector_args, lambdify_with_vector_args, grad

DEFAULT_CODE_GENERATOR = lambdify_with_vector_args
DEFAULT_CODE_GENERATOR_ARGS = {
    'modules': "numpy"
}

# TODO: A base System class? Enforces definition of dim_state, dim_input,
# dim_output, and functions before adding to BD (BD already does this for n's)
# and simulation (def needed to enforce functions) could even test dimensions
# of actual output to make sure its correct, but it will fail on sim w/o


class DynamicalSystem(object):
    def __init__(self, state_equation=None, state=None, input_=None,
                 output_equation=None, constants_values={}, dt=0,
                 initial_condition=None, code_generator=None,
                 code_generator_args={}):

        """
        state_equation is a vector valued expression, the derivative of the
        state.

        state is a sympy matrix (vector) of the state components, in desired
        order, matching state_equation.

        input_ is a sympy matrix (vector) of the input vector, in desired order

        output_equation is a vector valued expression, the output of the
        system.

        needs a "set vars to ___ then do ___" function. Used for eq points,
        phase plane, etc could be a "with" context??

        keep a list of constants, too?
        check for input/output connection ? (there's a name for this)
        check for autonomous/time-varying?
        check for control affine?
        check for memory(less)? just use n-state

        """
        # TODO: when constant_values is set, update callables?
        self.constants_values = constants_values
        self.state = state
        self.initial_condition = initial_condition
        self.input = input_

        self.code_generator = code_generator or DEFAULT_CODE_GENERATOR

        code_gen_args_to_set = DEFAULT_CODE_GENERATOR_ARGS.copy()
        code_gen_args_to_set.update(code_generator_args)
        self.code_generator_args = code_gen_args_to_set

        self.state_equation = state_equation
        self.output_equation = output_equation

        self.dt = dt

        self.n_events = 0

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state is None:  # or other checks?
            state = sp.Matrix([])
        if isinstance(state, sp.Expr):
            state = sp.Matrix([state])
        self.dim_state = len(state)
        self._state = state

    @property
    def input(self):
        return self._inputs

    @input.setter
    def input(self, input_):
        if input_ is None:  # or other checks?
            input_ = sp.Matrix([])
        if isinstance(input_, sp.Expr):  # check it's a single dynamicsymbol?
            input_ = sp.Matrix([input_])
        self.dim_input = len(input_)
        self._inputs = input_

    @property
    def state_equation(self):
        return self._state_equation

    @state_equation.setter
    def state_equation(self, state_equation):
        if state_equation is None:  # or other checks?
            state_equation = sp.Matrix([])
        assert len(state_equation) == len(self.state)
        assert find_dynamicsymbols(state_equation) <= (
                set(self.state) | set(self.input)
               )
        assert state_equation.atoms(sp.Symbol) <= (
                set(self.constants_values.keys()) | set([dynamicsymbols._t])
               )

        self._state_equation = state_equation
        self.update_state_equation_function()

        self.state_jacobian_equation = grad(self.state_equation, self.state)
        self.update_state_jacobian_function()

        self.input_jacobian_equation = grad(self.state_equation, self.input)
        self.update_input_jacobian_function()

    @property
    def output_equation(self):
        return self._output_equation

    @output_equation.setter
    def output_equation(self, output_equation):
        if output_equation is None:  # or other checks?
            output_equation = self.state
        try:
            self.dim_output = len(output_equation)
        except TypeError:
            self.dim_output = 1
        self._output_equation = output_equation
        assert output_equation.atoms(sp.Symbol) <= (
                set(self.constants_values.keys()) | set([dynamicsymbols._t])
               )
        if self.dim_state:
            assert find_dynamicsymbols(output_equation) <= set(self.state)
        else:
            assert find_dynamicsymbols(output_equation) <= set(self.input)
        self.update_output_equation_function()

    def update_state_equation_function(self):
        if not self.dim_state:
            return
        self.state_equation_function = self.code_generator(
            [dynamicsymbols._t] + sp.flatten(self.state) +
            sp.flatten(self.input),
            self.state_equation.subs(self.constants_values),
            **self.code_generator_args
        )

    def update_state_jacobian_function(self):
        if not self.dim_state:
            return
        self.state_jacobian_equation_function = self.code_generator(
            [dynamicsymbols._t] + sp.flatten(self.state) +
            sp.flatten(self.input),
            self.state_jacobian_equation.subs(self.constants_values),
            **self.code_generator_args
        )

    def update_input_jacobian_function(self):
        # TODO: state-less systems should have an input/output jacobian
        if not self.dim_state:
            return
        self.input_jacobian_equation_function = self.code_generator(
            [dynamicsymbols._t] + sp.flatten(self.state) +
            sp.flatten(self.input),
            self.input_jacobian_equation.subs(self.constants_values),
            **self.code_generator_args
        )

    def update_output_equation_function(self):
        if not self.dim_output:
            return
        if self.dim_state:
            self.output_equation_function = self.code_generator(
                [dynamicsymbols._t] + sp.flatten(self.state),
                self.output_equation.subs(self.constants_values),
                **self.code_generator_args
            )
        else:
            self.output_equation_function = self.code_generator(
                [dynamicsymbols._t] + sp.flatten(self.input),
                self.output_equation.subs(self.constants_values),
                **self.code_generator_args
            )

    @property
    def initial_condition(self):
        return self._initial_condition

    @initial_condition.setter
    def initial_condition(self, initial_condition):
        if initial_condition is not None:
            assert len(initial_condition) == self.dim_state
            self._initial_condition = initial_condition
        else:
            self._initial_condition = np.zeros(self.dim_state)

    def prepare_to_integrate(self):
        pass

    def copy(self):
        copy = self.__class__(
            state_equation=self.state_equation,
            state=self.state,
            input_=self.input,
            output_equation=self.output_equation,
            constants_values=self.constants_values,
            dt=self.dt
        )
        copy.output_equation_function = self.output_equation_function
        copy.state_equation_function = self.state_equation_function
        return copy

    def equilibrium_points(self, input_=None):
        return sp.solve(self.state_equation, self.state, dict=True)


class MemorylessSystem(DynamicalSystem):
    """
    a system with no state

    if no input are used, can represent a signal (function of time only)
    for example, a stochastic signal could interpolate points and use
    prepare_to_integrate to re-seed the data, or something.

    when I decouple code generator, maybe output_equation could even be a
    stochastic representation?
    """
    def __init__(self, input_=None, output_equation=None, **kwargs):
        super().__init__(
              input_=input_, output_equation=output_equation, **kwargs)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state is None:  # or other checks?
            state = sp.Matrix([])
        else:
            raise ValueError("Memoryless system should not have state or " +
                             "state_equation")
        self.dim_state = len(state)
        self._state = state


def SystemFromCallable(incallable, dim_input, dim_output, dt=0):
    system = MemorylessSystem(dt=dt)
    system.dim_input = dim_input
    system.dim_output = dim_output
    system.output_equation_function = incallable
    return system


class LTISystem(DynamicalSystem):
    def __init__(self, *args, constants_values={}, dt=0):
        """
        Pass in ABC/FGH matrices
        x' = Fx+Gu
        y = Hx

        or for a memoryless linear system (aka, state feedback), pass in
        K/D matrix
        y = Ku

        just wrappers for jacobian equations/functions?
        need to decide how to use symbolic vs numeric


        possible features:
            - hold symbolic structured matrices (0's where appropriate)
            - functions to convert between different canonical forms
            - stability analysis, controlability, observability, etc
            - discretize, z-transform
            - frequency response analysis
            - nyquist, root locus, etc
        """
        super().__init__(constants_values=constants_values, dt=dt)

        if len(args) not in (1, 2, 3):
            raise ValueError("LTI system expects 1, 2, or 3 args")

        # TODO: setup jacobian functions
        if len(args) == 1:
            self.K = K = args[0]
            self.dim_input = self.K.shape[1]
            self.dim_output = self.K.shape[0]
            self.output_equation_function = lambda t, x: (K@x).reshape(-1)
            return

        if len(args) == 2:
            F, G = args
            H = np.matlib.eye(F.shape[0])

        elif len(args) == 3:
            F, G, H = args

        self.F = np.asmatrix(F)
        self.G = np.asmatrix(G)
        self.H = np.asmatrix(H)

        self.dim_state = F.shape[0]
        self.dim_input = G.shape[1]
        self.dim_output = H.shape[0]
        self.state_equation_function = lambda t, x, u: (F@x + G@u).reshape(-1)
        self.output_equation_function = lambda t,x: (H*x).reshape(-1)


