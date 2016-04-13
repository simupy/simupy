import sympy as sp, numpy as np
from sympy.physics.mechanics import dynamicsymbols
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

        output_equations is avector valued expression, the output of the system.

        needs a "set vars to ___ then do ___" function. Used for eq points, phase plane, etc
        could be a "with" context??

        keep a list of constants, too?
        check for input/output connection ? (there's a name for this)
        check for autonomous/time-varying?
        check for control affine?
        check for memory(less)? just use n-state

        """
        self.constants_values = constants_values
        self.states = states
        self.inputs = inputs
        self.state_equations = state_equations
        self.output_equations = output_equations

        if initial_condition is not None:
            self.initial_condition = initial_condition
        else:
            self.initial_condition = np.zeros(self.n_states)
        
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
        self._state_equations = state_equations
        # TODO: decouple code generator. Perhaps allow  PyDy and/or PyODEsys
        self.state_equation_function = lambdify_with_vector_args( \
            [dynamicsymbols._t] + sp.flatten(self.states) + sp.flatten(self.inputs), \
            state_equations.subs(self.constants_values), modules="numpy")

    @property
    def output_equations(self):
        return self._output_equations

    @output_equations.setter
    def output_equations(self,output_equations):
        if output_equations is None: # or other checks?
            output_equations = self.states
        self.n_outputs = len(output_equations)
        self._output_equations = output_equations

        # TODO: decouple code generator. Perhaps allow  PyDy and/or PyODEsys
        if self.n_states:
            self.output_equation_function = lambdify_with_vector_args( \
                [dynamicsymbols._t] + sp.flatten(self.states), \
                output_equations.subs(self.constants_values), modules="numpy")
        else:
            self.output_equation_function = lambdify_with_vector_args( \
                [dynamicsymbols._t] + sp.flatten(self.inputs), \
                output_equations.subs(self.constants_values), modules="numpy")

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

    def jacobian(self):
        return grad(self.state_equations, self.states)

    def equilibrium_points(self,inputs=None):
        return sp.solve(self.state_equations, self.states, dict=True)

class DescriptorSystem(DynamicalSystem):
    # ideally, take advantage of DAE solvers eventually
    # since I'm on my own, I will use my generalized momentum nomenclature
    def __init__(self):

        self.mass_matrix = kwargs.pop('mass_matrix',None)
        if self.mass_matrix is not None:
            self.state_eq_explicit = kwargs.pop('state_eq_explicit', None)
            if self.state_eq_explicit is None:
                self.state_eq_explicit = self.mass_matrix.LUsolve(self.state_equations)
        else:
            self.state_eq_explicit = self.state_equations

class MemorylessSystem(DynamicalSystem):
    # a system with no states
    # if no inputs are used, can represent a signal
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
        """
        pass
