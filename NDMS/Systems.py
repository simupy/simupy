import sympy as sp, numpy as np
from .utils import process_vector_args, lambdify_with_vector_args, grad

class DynamicalSystem(object): 
    def __init__(self, state_equations, states, inputs=sp.Matrix([]), constants_values={}, dt=None):
        """
        state_equations is a vector valued expression, the derivative of each state.        

        states is a sympy matrix (vector) of the states, in desired order, matching 
        state_equations.

        TODO: This should really have output equations (functions)

        var name and description? (machine/human name)

        needs a "set vars to ___ then do ___" function. Used for eq points, phase plane, etc
        could be a "with" context??

        handle non-autonomous
        mappings for call function
        a "prepare to integrate" to do CSE for performance? later!

        """
        n_states, one_test = states.shape
        n_inputs, one_test = inputs.shape 
        n_states_test = len(state_equations)
        
        self.state_equations = state_equations
        self.states = states
        self.inputs = inputs
        self.constants_values = constants_values
        
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.dt = dt
            
        self.callable_function = lambdify_with_vector_args(sp.flatten(states) + sp.flatten(inputs), \
            state_equations.subs(constants_values), modules="numpy")
    
    def copy(self):
        return self.__class__(self.state_equations, self.states, self.inputs, self.constants_values, self.dt)

    def __call__(self,*args):
        return self.callable_function(*args)

    def jacobian(self):
        return grad(self.state_equations, self.states)

    def equilibrium_points(self,inputs=None):
        return sp.solve(self.state_equations, self.states, dict=True)

class SuperSystem(DynamicalSystem):
    """
    System with constructor for connecting DynamicalSystems, Controllers, Outputs, Observers, etc.
    Or should Systems just have operators? Like >> to connect, +/-/* for add/subtract/scalar multiply?
    That would certainly be cute!

    dictionary mapping system.state (var) to actual vector of data? similarly if it's just a single
    system, should have state/var mapping

    ideally something like connect(system1.output, system2.input)
    and be able to do things like connect(system1.output * system2.output, system3.input)
    and feedback might be: connect(system1.output, controller1.input), connect(controller1.output, system1.input)

    a stochastic simulation? generate random trajectory over entire simulation window. that means
    this would need to say when something is stochastic. 

    an expensive controller should also have memory; the controller should be responsible not the integrator

    This is really a BlockDiagram class, and is connected to simulation. This should setup simulation, generate any new stochastic signals as needed,
    etc, then run integration, and collect the resulting trajectories/signals.

    Is there a flag for vectorized?
    """
    pass

class LinearSystem(DynamicalSystem):
    pass

class MatrixSystem(DynamicalSystem):
    pass