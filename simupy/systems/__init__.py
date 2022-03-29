import numpy as np
from simupy.block_diagram import SimulationMixin
import warnings

need_state_equation_function_msg = ("if dim_state > 0, DynamicalSystem must"
                                    + " have a state_equation_function")

need_output_equation_function_msg = ("if dim_state == 0, DynamicalSystem must"
                                     + " have an output_equation_function")

zero_dim_output_msg = "A DynamicalSystem must provide an output"


def full_state_output(t, x, *args):
    """
    A drop-in ``output_equation_function`` for stateful systems that provide
    output the full state directly.
    """
    return x




class DynamicalSystem(SimulationMixin):
    """
    A dynamical system which models systems of the form::

        xdot(t) = state_equation_function(t,x,u)
        y(t) = output_equation_function(t,x)

    or::

        y(t) = output_equation_function(t,u)

    These could also represent discrete-time systems, in which case xdot(t)
    represents x[k+1].

    This can also model discontinuous systems. Discontinuities must occur on
    zero-crossings of the ``event_equation_function``, which take the same
    arguments as ``output_equation_function``, depending on ``dim_state``.
    At the zero-crossing, ``update_equation_function`` is called with the same
    arguments. If ``dim_state`` > 0, the return value of
    ``update_equation_function`` is used as the state of the system immediately
    after the discontinuity.
    """
    def __init__(self, state_equation_function=None,
                 output_equation_function=None, event_equation_function=None,
                 update_equation_function=None, dim_state=0, dim_input=0,
                 dim_output=0, num_events=0, dt=0, initial_condition=None):
        """
        Parameters
        ----------
        state_equation_function : callable, optional
            The derivative (or update equation) of the system state. Not needed
            if ``dim_state`` is zero.
        output_equation_function : callable, optional
            The output equation of the system. A system must have an
            ``output_equation_function``. If not set, uses full state output.
        event_equation_function : callable, optional
            The function whose output determines when discontinuities occur.
        update_equation_function : callable, optional
            The function called when a discontinuity occurs.
        dim_state : int, optional
            Dimension of the system state. Optional, defaults to 0.
        dim_input : int, optional
            Dimension of the system input. Optional, defaults to 0.
        dim_output : int, optional
            Dimension of the system output. Optional, defaults to dim_state.
        num_events : int, optional
            Dimension of the system event functions. Optional, defaults to 0.
        dt : float, optional
            Sample rate of the system. Optional, defaults to 0 representing a
            continuous time system. 
        initial_condition : array_like of numerical values, optional
            Array or Matrix used as the initial condition of the system.
            Defaults to zeros of the same dimension as the state.
        """
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dim_output = dim_output or dim_state
        self.num_events = num_events


        self.state_equation_function = state_equation_function

        self.output_equation_function = (
            full_state_output
            if output_equation_function is None and self.dim_state > 0
            else output_equation_function
        )

        self.initial_condition = initial_condition

        if ((num_events != 0) and ((event_equation_function is None) or
        (update_equation_function is None))):
            raise ValueError("Cannot provide event_equation_function or " + 
                             "update_Equation_function with num_events == 0")

        self.event_equation_function = event_equation_function
        # TODO: do some defensive checks and/or wrapping of update function to consume
        # a channel number
        self.update_equation_function = update_equation_function

        self.dt = dt

        self.validate()

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        if dt <= 0:
            self._dt = 0
            return
        if self.num_events != 0:
            raise ValueError("Cannot set dt > 0 and use event API " +
                             "with non-zero num_events")
        self.num_events = 1
        self._dt = dt
        self.event_equation_function = lambda t, *args: np.atleast_1d(np.sin(np.pi*t/self.dt))
        #    if t else np.sin(np.finfo(np.float_).eps))
        self._state_equation_function = self.state_equation_function
        self._output_equation_function = self.output_equation_function
        self.state_equation_function = \
            lambda *args: np.zeros(self.dim_state)
        if self.dim_state:
            self.update_equation_function = (
                lambda *args, event_channels=0: self._state_equation_function(*args)
            )
        else:
            self._prev_output = 0
            def _update_equation_function(*args, event_channels=0):
                self._prev_output = self._output_equation_function(*args)

            self.update_equation_function = _update_equation_function
            self.output_equation_function = lambda *args: self._prev_output


    @property
    def initial_condition(self):
        if self._initial_condition is None:
            self._initial_condition = np.zeros(self.dim_state)
        return self._initial_condition

    @initial_condition.setter
    def initial_condition(self, initial_condition):
        if initial_condition is not None:
            if isinstance(initial_condition, np.ndarray):
                size = initial_condition.size
            else:
                size = len(initial_condition)
            assert size == self.dim_state
            self._initial_condition = np.array(initial_condition,
                dtype=np.float_).reshape(-1)
        else:
            self._initial_condition = None

    def prepare_to_integrate(self, t0, state_or_input=None):
        # if dim_state >0, the initial state should have all information required to
        # simulate; for state-less systems, call update equation as default
        # initialization
        if not self.dim_state and self.num_events:
            if self.dim_input:
                self.update_equation_function(t0, state_or_input)
            else:
                self.update_equation_function(t0,)
        # regardless, 
        if self.dim_state or self.dim_input:
            return self.output_equation_function(t0, state_or_input)
        else:
            return self.output_equation_function(t0)

    def validate(self):
        if self.dim_output == 0:
            raise ValueError(zero_dim_output_msg)

        if (self.dim_state > 0
                and getattr(self, 'state_equation_function', None) is None):
            raise ValueError(need_state_equation_function_msg)

        if (self.dim_state == 0
                and getattr(self, 'output_equation_function', None) is None):
            raise ValueError(need_output_equation_function_msg)


def SystemFromCallable(incallable, dim_input, dim_output, dt=0):
    """
    Construct a memoryless system from a callable.

    Parameters
    ----------
    incallable : callable
        Function to use as the output_equation_function. Should have signature
        (t, u) if dim_input > 0 or (t) if dim_input = 0.
    dim_input : int
        Dimension of input.
    dim_output : int
        Dimension of output.
    """
    system = DynamicalSystem(output_equation_function=incallable,
                             dim_input=dim_input, dim_output=dim_output, dt=dt)
    return system


class SwitchedSystem(DynamicalSystem):
    """
    Provides a useful pattern for discontinuous systems where the state and
    output equations change depending on the value of a function of the state
    and/or input (``event_variable_equation_function``). Most of the usefulness
    comes from constructing the ``event_equation_function`` with a Bernstein
    basis polynomial with roots at the boundaries. This class also provides
    logic for outputting the correct state and output equation based on the
    ``event_variable_equation_function`` value.
    """
    def __init__(self, state_equations_functions=None,
                 output_equations_functions=None,
                 event_variable_equation_function=None, event_bounds=None,
                 state_update_equation_function=None, dim_state=0, dim_input=0,
                 dim_output=0, initial_condition=None):
        """
        Parameters
        ----------
        state_equations_functions : array_like of callables, optional
            The derivative (or update equation) of the system state. Not needed
            if ``dim_state`` is zero. The array indexes the
            event-state and should be one more than the number of event bounds.
            This should also be indexed to match the boundaries (i.e., the
            first function is used when the event variable is below the first
            event_bounds value). If only one callable is provided, the callable
            is used in each condition.
        output_equations_functions : array_like of callables, optional
            The output equation of the system. A system must have an
            ``output_equation_function``. If not set, uses full state output.
            The array indexes the event-state and should be one more than the
            number of event bounds. This should also be indexed to match the
            boundaries (i.e., the first function is used when the event
            variable is below the first event_bounds value). If only one
            callable is provided, the callable is used in each condition.
        event_variable_equation_function : callable
            When the output of this function crosses the values in
            ``event_bounds``, a discontuity event occurs.
        event_bounds : array_like of floats
            Defines the boundary points the trigger discontinuity events based
            on the output of ``event_variable_equation_function``.
        state_update_equation_function : callable, optional
            When an event occurs, the state update equation function is called
            to determine the state update. If not set, uses full state output,
            so the state is not changed upon a zero-crossing of the event
            variable function.
        dim_state : int, optional
            Dimension of the system state. Optional, defaults to 0.
        dim_input : int, optional
            Dimension of the system input. Optional, defaults to 0.
        dim_output : int, optional
            Dimension of the system output. Optional, defaults to dim_state.
        """
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dim_output = dim_output or dim_state
        self.event_bounds = event_bounds

        self.state_equations_functions = np.empty(self.n_conditions,
                                                  dtype=object)
        self.state_equations_functions[:] = state_equations_functions

        self.output_equations_functions = np.empty(self.n_conditions,
                                                   dtype=object)
        self.output_equations_functions[:] = (
            full_state_output
            if output_equations_functions is None and self.dim_state > 0
            else output_equations_functions
        )

        self.event_variable_equation_function = \
            event_variable_equation_function

        self.state_update_equation_function = (
            state_update_equation_function or
            full_state_output
        )

        self.initial_condition = initial_condition
        self.dt = 0

        self.validate()

    def validate(self):
        super().validate()

        if (self.dim_state > 0
                and np.any(np.equal(self.state_equations_functions, None))):
            raise ValueError(need_state_equation_function_msg)

        if (self.dim_state == 0
                and np.any(np.equal(self.output_equations_functions, None))):
            raise ValueError(need_output_equation_function_msg)

        if self.event_variable_equation_function is None:
            raise ValueError("A SwitchedSystem requires " +
                             "event_variable_equation_function")

    @property
    def event_bounds(self):
        return self._event_bounds

    @event_bounds.setter
    def event_bounds(self, event_bounds):
        if event_bounds is None:
            raise ValueError("A SwitchedSystem requires event_bounds")
        self._event_bounds = np.array(event_bounds).reshape(1, -1)
        self.n_conditions = self._event_bounds.size + 1
        if self.n_conditions == 2:
            self.event_bounds_range = 1
        else:
            self.event_bounds_range = np.diff(self.event_bounds[0, [0, -1]])

    def output_equation_function(self, *args):
        return self.output_equations_functions[self.condition_idx](*args)

    def state_equation_function(self, *args):
        return self.state_equations_functions[self.condition_idx](*args)

    def event_equation_function(self, *args):
        event_var = self.event_variable_equation_function(*args)
        return np.prod(
            (self.event_bounds_range-self.event_bounds)*event_var -
            self.event_bounds*(self.event_bounds_range - event_var),
            axis=1
        )

    def update_equation_function(self, *args):
        event_var = self.event_variable_equation_function(*args)
        if self.condition_idx is None: 
            self.condition_idx = np.where(np.all(np.r_[
                    np.c_[[[True]], event_var >= self.event_bounds],
                    np.c_[event_var <= self.event_bounds, [[True]]]
                    ], axis=0))[0][0]
            return
        sq_dist = (event_var - self.event_bounds)**2
        crossed_root_idx = np.where(sq_dist == np.min(sq_dist))[1][0]
        if crossed_root_idx == self.condition_idx:
            self.condition_idx += 1
        elif crossed_root_idx == self.condition_idx-1:
            self.condition_idx -= 1
        else:
            warnings.warn("SwitchedSystem did not cross a neighboring " +
                          "boundary. This may indicate an integration " +
                          "error. Continuing without updating " +
                          "condition_idx", UserWarning)
        return self.state_update_equation_function(*args)

    def prepare_to_integrate(self):
        if self.dim_state:
            event_var = self.event_variable_equation_function(0, 
                self.initial_condition)
            self.condition_idx = np.where(np.all(np.r_[
                    np.c_[[[True]], event_var >= self.event_bounds],
                    np.c_[event_var <= self.event_bounds, [[True]]]
                    ], axis=0))[0][0]
        else:
            self.condition_idx = None


class LTISystem(DynamicalSystem):
    """
    A linear, time-invariant system.
    """
    def __init__(self, *args, initial_condition=None, dt=0):
        """
        Construct an LTI system with the following input formats:

        1. state matrix A, input matrix B, output matrix C for systems with
           state::

              dx_dt = Ax + Bu
              y = Hx

        2. state matrix A, input matrix B for systems with state, assume full
           state output::

              dx_dt = Ax + Bu
              y = Ix

        3. gain matrix K for systems without state::

              y = Kx


        The matrices should be numeric arrays of consistent shape. The class
        provides ``A``, ``B``, ``C`` and ``F``, ``G``, ``H`` aliases for the
        matrices of systems with state, as well as a ``K`` alias for the gain
        matrix. The ``data`` alias provides the matrices as a tuple.
        """

        if len(args) not in (1, 2, 3):
            raise ValueError("LTI system expects 1, 2, or 3 args")

        self.num_events = 0
        self.event_equation_function = None
        self.update_equation_function = None

        # TODO: setup jacobian functions
        if len(args) == 1:
            self.gain_matrix = gain_matrix = np.array(args[0])
            self.dim_input = (self.gain_matrix.shape[1]
                              if len(gain_matrix.shape) > 1
                              else 1)
            self.dim_output = self.gain_matrix.shape[0]
            self.dim_state = 0
            self.initial_condition = None
            self.state_equation_function = None
            self.output_equation_function = \
                lambda t, x: (gain_matrix@x).reshape(-1)
            self.dt = dt
            return

        if len(args) == 2:
            state_matrix, input_matrix = args
            output_matrix = np.eye(
                getattr(state_matrix, 'shape', len(state_matrix))[0]
            )

        elif len(args) == 3:
            state_matrix, input_matrix, output_matrix = args

        if len(input_matrix.shape) == 1:
            input_matrix = input_matrix.reshape(-1, 1)

        state_matrix = np.array(state_matrix)
        input_matrix = np.array(input_matrix)
        output_matrix = np.array(output_matrix)

        self.dim_state = state_matrix.shape[0]
        self.dim_input = input_matrix.shape[1]
        self.dim_output = output_matrix.shape[0]

        self.state_matrix = state_matrix
        self.input_matrix = input_matrix
        self.output_matrix = output_matrix

        self.initial_condition = initial_condition
        if self.dim_input:
            self.state_equation_function = \
                (lambda t, x, u=np.zeros(self.dim_input): \
                    (state_matrix@x + input_matrix@u))
        else:
            self.state_equation_function = lambda t, x, u=np.zeros(0): state_matrix@x

        self.output_equation_function = \
            lambda t, x: (output_matrix@x)

        self.dt = dt

        self.validate()

    def validate(self):
        super().validate()
        if self.dim_state:
            assert self.state_matrix.shape[1] == self.dim_state
            assert self.input_matrix.shape[0] == self.dim_state
            assert self.output_matrix.shape[1] == self.dim_state

    @property
    def data(self):
        if self.dim_state:
            return self.state_matrix, self.input_matrix, self.output_matrix
        else:
            return self.gain_matrix

    @property
    def A(self):
        return self.state_matrix

    @property
    def F(self):
        return self.state_matrix

    @property
    def B(self):
        return self.input_matrix

    @property
    def G(self):
        return self.input_matrix

    @property
    def C(self):
        return self.output_matrix

    @property
    def H(self):
        return self.output_matrix

    @property
    def K(self):
        return self.gain_matrix
