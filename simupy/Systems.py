import sympy as sp, numpy as np
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics.functions import find_dynamicsymbols
from .utils import process_vector_args, lambdify_with_vector_args, grad

class DynamicalSystem(object): 
    def __init__(self, state_equations=None, states=None, inputs=None, 
            output_equations=None, constants_values={}, dt=0, 
            initial_condition=None):

        """
        state_equations is a vector valued expression, the derivative of each state.        

        states is a sympy matrix (vector) of the states, in desired order, matching 
        state_equations.

        inputs is a sympy matrix (vector) of the inputs, in desired order

        output_equations is a vector valued expression, the output of the system.

        needs a "set vars to ___ then do ___" function. Used for eq points, phase plane, etc
        could be a "with" context??

        keep a list of constants, too?
        check for input/output connection ? (there's a name for this)
        check for autonomous/time-varying?
        check for control affine?
        check for memory(less)? just use n-state

        """
        # TODO: when constant_values is set, update callables?
        self.constants_values = constants_values
        self.states = states
        self.initial_condition = initial_condition
        self.inputs = inputs
        self.state_equations = state_equations
        self.output_equations = output_equations
        
        self.dt = dt

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self,states):
        if states is None: # or other checks?
            states = sp.Matrix([])
        self.n_states = len(states)
        self._states = states

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self,inputs):
        if inputs is None: # or other checks?
            inputs = sp.Matrix([])
        self.n_inputs = len(inputs)
        self._inputs = inputs

    @property
    def state_equations(self):
        return self._state_equations

    @state_equations.setter
    def state_equations(self,state_equations):
        if state_equations is None: # or other checks?
            state_equations = sp.Matrix([])
        assert len(state_equations) == len(self.states)
        assert find_dynamicsymbols(state_equations) <= set(self.states) | set(self.inputs)
        assert state_equations.atoms(sp.Symbol) <= set(self.constants_values.keys()) | set([dynamicsymbols._t])
        self._state_equations = state_equations
        # TODO: decouple code generator. Perhaps allow  PyDy and/or PyODEsys
        self.state_equation_function = lambdify_with_vector_args( \
            [dynamicsymbols._t] + sp.flatten(self.states) + sp.flatten(self.inputs), \
            state_equations.subs(self.constants_values), modules="numpy")

        self.state_jacobian_equation = grad(self.state_equations, self.states)
        self.state_jacobian_equation_function = lambdify_with_vector_args( \
            [dynamicsymbols._t] + sp.flatten(self.states) + sp.flatten(self.inputs), \
            self.state_jacobian_equation.subs(self.constants_values), modules="numpy")

        self.input_jacobian_equation = grad(self.state_equations, self.inputs)
        self.input_jacobian_equation_function = lambdify_with_vector_args( \
            [dynamicsymbols._t] + sp.flatten(self.states) + sp.flatten(self.inputs), \
            self.input_jacobian_equation.subs(self.constants_values), modules="numpy")

    @property
    def output_equations(self):
        return self._output_equations

    @output_equations.setter
    def output_equations(self,output_equations):
        if output_equations is None: # or other checks?
            output_equations = self.states
        self.n_outputs = len(output_equations)
        self._output_equations = output_equations
        assert output_equations.atoms(sp.Symbol) <= set(self.constants_values.keys()) | set([dynamicsymbols._t])
        
        if self.n_states:
            assert find_dynamicsymbols(output_equations) <= set(self.states)

            # TODO: decouple code generator. Perhaps allow  PyDy and/or PyODEsys
            self.output_equation_function = lambdify_with_vector_args( \
                [dynamicsymbols._t] + sp.flatten(self.states), \
                output_equations.subs(self.constants_values), modules="numpy")
        else:
            assert find_dynamicsymbols(output_equations) <= set(self.inputs)

            # TODO: decouple code generator. Perhaps allow  PyDy and/or PyODEsys
            self.output_equation_function = lambdify_with_vector_args( \
                [dynamicsymbols._t] + sp.flatten(self.inputs), \
                output_equations.subs(self.constants_values), modules="numpy")

    @property
    def initial_condition(self):
        return self._initial_condition

    @initial_condition.setter
    def initial_condition(self,initial_condition):
        if initial_condition is not None:
            assert len(initial_condition) == self.n_states
            self._initial_condition = initial_condition
        else:
            self._initial_condition = np.zeros(self.n_states)

    def prepare_to_integrate(self):
        pass

    def copy(self):
        copy = self.__class__(state_equations=self.state_equations, \
            states=self.states, inputs=self.inputs, \
            output_equations=self.output_equations, 
            constants_values=self.constants_values, dt=self.dt)
        copy.output_equation_function = self.output_equation_function
        copy.state_equation_function = self.state_equation_function
        return copy

    def equilibrium_points(self,inputs=None):
        return sp.solve(self.state_equations, self.states, dict=True)

class DescriptorSystem(DynamicalSystem):
    # ideally, take advantage of DAE solvers eventually
    # since I'm on my own, I will use my generalized momentum nomenclature
    def __init__(self, mass_matrix=None, impulse_equations=None, states=None, 
        inputs=None, output_equations=None, constants_values={}, dt=0, 
        initial_condition=None):

        self.constants_values = constants_values
        self.states = states
        self.initial_condition = initial_condition
        self.inputs = inputs
        self.output_equations = output_equations
        self.impulse_equations = impulse_equations
        self.mass_matrix = mass_matrix
        self.dt = dt

    @property 
    def impulse_equations(self):
        return self._impulse_equations

    @impulse_equations.setter
    def impulse_equations(self, impulse_equations):
        assert find_dynamicsymbols(impulse_equations) <= set(self.states) | set(self.inputs)
        assert impulse_equations.atoms(sp.Symbol) <= set(self.constants_values.keys()) | set([dynamicsymbols._t])
        self._impulse_equations = impulse_equations

    @property
    def mass_matrix(self):
        return self._mass_matrix

    @mass_matrix.setter
    def mass_matrix(self,mass_matrix):
        if mass_matrix is None:
            mass_matrix = sp.eye(self.n_states)
        assert mass_matrix.shape[1] == len(self.states)
        assert mass_matrix.shape[0] == len(self.impulse_equations)
        assert find_dynamicsymbols(mass_matrix) <= set(self.states) | set(self.inputs)
        assert mass_matrix.atoms(sp.Symbol) <= set(self.constants_values.keys()) | set([dynamicsymbols._t])

        self.state_equations = mass_matrix.LUsolve(self.impulse_equations)
        self._mass_matrix = mass_matrix
        # TODO: callable for mass matrices and impulse_equations for DAE solvers

class MemorylessSystem(DynamicalSystem):
    """
    a system with no states
    
    if no inputs are used, can represent a signal (function of time only)
    for example, a stochastic signal could interpolate points and use 
    prepare_to_integrate to re-seed the data, or something.

    when I decouple code generator, maybe output_equations could even be a
    stochastic representation? 
    """
    def __init__(self, inputs=None, output_equations=None, constants_values={}, dt=0):
        super(MemorylessSystem,self).__init__(inputs=inputs,  output_equations=output_equations,
           constants_values=constants_values, dt=dt)

def SystemFromCallable(incallable,n_inputs,n_outputs,dt=0):
    system = MemorylessSystem(dt=dt)
    system.n_inputs = n_inputs
    system.n_outputs = n_outputs
    system.output_equation_function = incallable
    return system

class LTISystem(DynamicalSystem):
    def __init__(self, *args):
        """
        Pass in ABC/FGH matrices
        x' = Fx+Gu
        y = Hx

        or for a memoryless linear system (aka, state feedback), pass in K/D matrix
        y = Ku

        hold symbolic and/or numeric matrices, plus callables

        should have a generator that takes just the numerical, and creates the symbolic with the same
        states. can also do the standard, define symbolic matrices with constants
        """
        pass
