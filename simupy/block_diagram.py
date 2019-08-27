from scipy.integrate import ode
import numpy as np
import warnings
import sys as syslib
from simupy.utils import callable_from_trajectory
from scipy.optimize import brentq

DEFAULT_INTEGRATOR_CLASS = ode
DEFAULT_INTEGRATOR_OPTIONS = {
        'name': 'dopri5',
        'rtol': 1e-6,
        'atol': 1e-12,
        'nsteps': 500,
        'max_step': 0.0
    }

DEFAULT_EVENT_FINDER = brentq
DEFAULT_EVENT_FIND_OPTIONS = {
        'xtol': 2e-12,
        'rtol': 8.8817841970012523e-16,
        'maxiter': 100
    }

nan_warning_message = ("BlockDiagram encountered NaN outputs and quit during" +
                       " {}. This may have been intentional! NaN outputs at " +
                       "time t={}, state x={}, output y={}")


class SimulationResult(object):
    """
    A simple class to collect simulation result trajectories.

    Attributes
    ----------
    t : array of times
    x : array of states
    y : array of outputs
    e : array of events
    """

    max_allocation = 2**7

    def __init__(self, dim_states, dim_outputs, tspan, n_sys, initial_size=0):
        if initial_size == 0:
            initial_size = tspan.size
        self.t = np.empty(initial_size)
        self.x = np.empty((initial_size, dim_states))
        self.y = np.empty((initial_size, dim_outputs))
        self.e = np.empty((initial_size, n_sys))
        self.res_idx = 0
        self.tspan = tspan
        self.t0 = tspan[0]
        self.tF = tspan[-1]

    def allocate_space(self, t):
        more_rows = int((self.tF-t)*self.t.size/(t-self.t0))+1
        more_rows = max(min(more_rows, self.max_allocation),1)

        self.t = np.r_[self.t, np.empty(more_rows)]
        self.x = np.r_[self.x, np.empty((more_rows, self.x.shape[1]))]
        self.y = np.r_[self.y, np.empty((more_rows, self.y.shape[1]))]
        self.e = np.r_[self.e, np.empty((more_rows, self.e.shape[1]))]

    def new_result(self, t, x, y, e=None):
        if self.res_idx >= self.t.size:
            self.allocate_space(t)
        self.t[self.res_idx] = t
        self.x[self.res_idx, :] = x
        self.y[self.res_idx, :] = y
        if e is not None:
            self.e[self.res_idx, :] = e
        else:
            self.e[self.res_idx, :] = np.zeros(self.e.shape[1])
        self.res_idx += 1

    def last_result(self, n=1, copy=False):
        n = np.clip(n, 1, self.res_idx)
        if copy:
            return (np.copy(self.t[self.res_idx-n]),
                    np.copy(self.x[self.res_idx-n, :]),
                    np.copy(self.y[self.res_idx-n, :]))
        else:
            return (self.t[self.res_idx-n], self.x[self.res_idx-n, :],
                    self.y[self.res_idx-n, :])


class BlockDiagram(object):
    """
    A block diagram of dynamical systems with their connections which can be
    numerically simulated.
    """

    def __init__(self, *systems):
        """
        Initialize a BlockDiagram, with an optional list of systems to start
        the diagram.
        """
        self.systems = np.array([], dtype=object)
        self.connections = np.array([], dtype=np.bool_).reshape((0, 0))

        self.dts = np.array([], dtype=np.float_)
        self.events = np.array([], dtype=np.bool_)
        self.cum_inputs = np.array([0], dtype=np.int_)
        self.cum_outputs = np.array([0], dtype=np.int_)
        self.cum_states = np.array([0], dtype=np.int_)
        self.cum_events = np.array([0], dtype=np.int_)

        self.inputs = np.array([], dtype=np.bool_).reshape((0,0))
        self.dim_input = 0

        for sys in systems:
            self.add_system(sys)
    
    @property
    def initial_condition(self):
        x0 = np.zeros(self.cum_states[-1])  # TODO: pre-allocate?
        for sysidx in np.where(self.systems)[0]:
            sys = self.systems[sysidx]
            state_start = self.cum_states[sysidx]
            state_end = self.cum_states[sysidx+1]
            x0[state_start:state_end] = sys.initial_condition
        return x0

    @property
    def dim_state(self):
        return self.cum_states[-1]

    @property
    def dim_output(self):
        # TODO: allow internal outputs to be "closed"? For now, no
        return self.cum_outputs[-1]

    @property
    def dt(self):
        return self.dts.min()
    
    def prepare_to_integrate(self):
        for sys in self.systems:
            sys.prepare_to_integrate()

        if np.sum(self.events) > 0:
            self.event_equation_function = self.event_equation_function_implementation
            self.update_equation_function = self.update_equation_function_implementation
        else:
            self.event_equation_function = None
            self.update_equation_function = None

    def create_input(self, to_system_input, channels=[], inputs=[]):
        """
        Create or use input channels to use block diagram as a subsystem.

        Parameters
        ----------
        channels : list-like
            Selector index of the input channels to connect.
        to_system_input : dynamical system
            The system (already added to BlockDiagram) to which inputs will be
            connected. Note that any previous input connections will be
            over-written.
        inputs : list-like, optional
            Selector index of the inputs to connect. If not specified or of
            length 0, will connect all of the inputs.
        """
        channels = np.asarray(channels)
        if len(channels) == 0:
            raise ValueError("Cannot create input without specifying channel")
        if np.min(channels) < 0:
            raise ValueError("Cannot create input channel < 0")

        if len(inputs) == 0:
            inputs = np.arange(to_system_input.dim_input)
        else:
            inputs = np.asarray(inputs)
        inputs = inputs + self.cum_inputs[
                      np.where(self.systems == to_system_input)
                  ]

        if len(channels) != len(inputs) and len(channels) != 1:
            raise ValueError("Cannot broadcast channels to inputs")

        if np.max(channels) > self.dim_input-1:
            self.inputs = np.pad(self.inputs,
                                    ((0, np.max(channels) - self.dim_input+1),
                                     (0, 0)),
                                    'constant', constant_values = 0)
            self.dim_input = np.max(channels)+1

        self.inputs[:, inputs] = False
        self.connections[:, inputs] = False

        self.inputs[channels, inputs] = True


    def connect(self, from_system_output, to_system_input, outputs=[], 
        inputs=[]):
        """
        Connect systems in the block diagram.

        Parameters
        ----------
        from_system_output : dynamical system
            The system (already added to BlockDiagram) from which outputs will
            be connected. Note that the outputs of a system can be connected to
            multiple inputs.
        to_system_input : dynamical system
            The system (already added to BlockDiagram) to which inputs will be
            connected. Note that any previous input connections will be
            over-written.
        outputs : list-like, optional
            Selector index of the outputs to connect. If not specified or of
            length 0, will connect all of the outputs.
        inputs : list-like, optional
            Selector index of the inputs to connect. If not specified or of
            length 0, will connect all of the inputs.
        """
        if len(outputs) == 0:
            outputs = np.arange(from_system_output.dim_output)
        else:
            outputs = np.asarray(outputs)
        outputs = outputs + self.cum_outputs[
                        np.where(self.systems == from_system_output)
                    ]

        if len(inputs) == 0:
            inputs = np.arange(to_system_input.dim_input)
        else:
            inputs = np.asarray(inputs)
        inputs = inputs + self.cum_inputs[
                      np.where(self.systems == to_system_input)
                  ]

        # TODO: Check that this can be broadcast correctly

        self.inputs[:, inputs] = False
        self.connections[:, inputs] = False

        self.connections[outputs, inputs] = True

    def add_system(self, system):
        """
        Add a system to the block diagram

        Parameters
        ----------
        system : dynamical system
            System to add to BlockDiagram
        """
        self.systems = np.append(self.systems, system)
        self.cum_states = np.append(self.cum_states,
                                    self.cum_states[-1] + system.dim_state)
        self.cum_inputs = np.append(self.cum_inputs,
                                    self.cum_inputs[-1] + system.dim_input)
        self.cum_outputs = np.append(self.cum_outputs,
                                     self.cum_outputs[-1] + system.dim_output)
        self.events = np.append(self.events, np.bool_(
                        getattr(system, 'event_equation_function', None) and
                        getattr(system, 'update_equation_function', None)))
        self.cum_events = np.append(self.cum_events,
                                    self.cum_events[-1] + self.events[-1])
        self.dts = np.append(self.dts, getattr(system, 'dt', 0))
        self.connections = np.pad(self.connections,
                                  ((0, system.dim_output),
                                   (0, system.dim_input)),
                                  'constant', constant_values=0)
        self.inputs = np.pad(self.inputs,
                                  ((0, 0),
                                   (0, system.dim_input)),
                                  'constant', constant_values=0)

    def output_equation_function(self, t, state, input_=None, update_memoryless_event=False):
        output = np.zeros(self.cum_outputs[-1])
        input_ = input_ if input_ is not None else np.zeros(self.dim_input)
        # compute outputs for full systems, y[t_k]=h(t_k,x[t_k])
        for sysidx in np.where((np.diff(self.cum_states) > 0))[0]:
            sys = self.systems[sysidx]
            output_start = self.cum_outputs[sysidx]
            output_end = self.cum_outputs[sysidx+1]
            state_start = self.cum_states[sysidx]
            state_end = self.cum_states[sysidx+1]

            state_values = state[state_start:state_end]
            output[output_start:output_end] = \
                sys.output_equation_function(t, state_values).reshape(-1)

        # compute outputs for memoryless systems, y[t_k]=h(t_k,u[t_k])
        for sysidx in np.where((np.diff(self.cum_states) == 0))[0][::-1]:
            sys = self.systems[sysidx]
            output_start = self.cum_outputs[sysidx]
            output_end = self.cum_outputs[sysidx+1]
            input_start = self.cum_inputs[sysidx]
            input_end = self.cum_inputs[sysidx+1]

            input_values = np.zeros(sys.dim_input)
            
            input_index, output_index = np.where(
                self.connections[:, input_start:input_end].T
            )
            input_values[input_index] = output[output_index]

            input_index, as_sys_input_index = np.where(
                self.inputs[:, input_start:input_end].T
            )
            
            if as_sys_input_index.size:
                input_values[input_index] = input_[as_sys_input_index]

            if sys.dim_input:
                if self.events[sysidx] and update_memoryless_event:
                    sys.update_equation_function(t, input_values)
                output[output_start:output_end] = \
                  sys.output_equation_function(t, input_values).reshape(-1)
            else:
                if self.events[sysidx] and update_memoryless_event:
                    sys.update_equation_function(t)
                output[output_start:output_end] = \
                  sys.output_equation_function(t).reshape(-1)

        return output

    def state_equation_function(self, t, state, input_=None, output=None):
        # TODO: how to define available inputs?? 
        dxdt = np.zeros(self.cum_states[-1])
        output = output if output is not None else self.output_equation_function(t, state, input_)

        for sysidx in np.where((np.diff(self.cum_states) > 0))[0]:
            sys = self.systems[sysidx]

            state_start = self.cum_states[sysidx]
            state_end = self.cum_states[sysidx+1]
            state_values = state[state_start:state_end]

            input_start = self.cum_inputs[sysidx]
            input_end = self.cum_inputs[sysidx+1]

            input_values = np.zeros(sys.dim_input)
            
            input_index, output_index = np.where(
                self.connections[:, input_start:input_end].T
            )
            input_values[input_index] = output[output_index]

            input_index, as_sys_input_index = np.where(
                self.inputs[:, input_start:input_end].T
            )
            
            if as_sys_input_index.size:
                input_values[input_index] = input_[as_sys_input_index]

            if sys.dim_input:
                dxdt[state_start:state_end] = \
                    sys.state_equation_function(
                        t, state_values, input_values
                    ).reshape(-1)
            else:
                dxdt[state_start:state_end] = \
                    sys.state_equation_function(t, state_values).reshape(-1)

        return dxdt

    def systems_event_equation_functions(self, t, state, output):
        events = np.zeros(self.systems.size)

        # compute events for stateful systems
        for sysidx in np.where(
                    (np.diff(self.cum_states) > 0) & self.events
                )[0]:
            sys = self.systems[sysidx]
            state_start = self.cum_states[sysidx]
            state_end = self.cum_states[sysidx+1]
            state_values = state[state_start:state_end]
            events[sysidx] = sys.event_equation_function(
                                t, state_values).reshape(-1)

        # compute events for memoryless systems
        for sysidx in np.where(
                    (np.diff(self.cum_states) == 0) & self.events
                )[0]:
            sys = self.systems[sysidx]
            input_start = self.cum_inputs[sysidx]
            input_end = self.cum_inputs[sysidx+1]

            input_values = np.zeros(sys.dim_input)
            
            input_index, output_index = np.where(
                self.connections[:, input_start:input_end].T
            )
            input_values[input_index] = output[output_index]

            if sys.dim_input:
                events[sysidx] = sys.event_equation_function(
                                        t, input_values).reshape(-1)
            else:
                events[sysidx] = sys.event_equation_function(
                                        t).reshape(-1)

        return events

    def event_equation_function_implementation(self, t, state, output=None):
        output = output or self.output_equation_function(t,state)
        return np.prod(
            self.systems_event_equation_functions(t, state, output))

    def update_equation_function_implementation(self, t, state, input_=None, output=None):
        next_state = state.copy()
        output = output or self.output_equation_function(t,state)
        input_ = input_ or np.zeros(self.dim_input)
        # find which one(s) crossed
        # call that/those systems's update_equation_function & fill in next_state
        return next_state

    def computation_step(self, t, state, output=None, selector=True, do_events=False):
        """
        callable to compute system outputs and state derivatives
        """
        # TODO: make sure this still works

        # TODO: p sure I just had output_equation_function here
        # I guess the outputs_in wasn't really necessary?
        output = output if output is not None else \
            self.output_equation_function(t, state)

        # compute state equation for full systems,
        # x[t_k']=f(t_k,x[t_k],u[t_k])
        dxdt = self.state_equation_function(t, state, output=output)
        
        if do_events:
            events = self.systems_event_equation_functions(t, state, output)

            return dxdt, output, events

        return dxdt, output

    def simulate(self, tspan, integrator_class=DEFAULT_INTEGRATOR_CLASS,
                 integrator_options=DEFAULT_INTEGRATOR_OPTIONS,
                 event_finder=DEFAULT_EVENT_FINDER,
                 event_find_options=DEFAULT_EVENT_FIND_OPTIONS):
        """
        Simulate the block diagram

        Parameters
        ----------
        tspan : list-like or float

            Argument to specify integration time-steps.

            If a single time is specified, it is treated as the final time.
            If two times are specified, they are treated as initial and
            final times. In either of these conditions, it is assumed that
            that every time step from a variable time-step integrator will
            be stored in the result.

            If more than two times are specified, these are the only times
            where the trajectories will be stored.

        integrator_class : class, optional
            Class of integrator to use. Defaults to ``scipy.integrate.ode``.
            Must provide the following subset of the ``scipy.integrate.ode``
            API:

                - ``__init__(derivative_callable(time, state))``
                - ``set_integrator(**kwargs)``
                - ``set_initial_value(state, time)``
                - ``set_solout(successful_step_callable(time, state))``
                - ``integrate(time)``
                - ``successful()``
                - ``y``, ``t`` properties

        integrator_options : dict, optional
            Dictionary of keyword arguments to pass to
            ``integrator_class.set_integrator``.
        event_finder : callable, optional
            Interval root-finder function. Defaults to
            ``scipy.optimize.brentq``, and must take the equivalent positional
            arguments, ``f``, ``a``, and ``b``, and return ``x0``, where
            ``a <= x0 <= b`` and ``f(x0)`` is the zero.
        event_find_options : dict, optional
            Dictionary of keyword arguments to pass to ``event_finder``. It
            must provide a key ``'xtol'``, and it is expected that the exact
            zero lies within ``x0 +/- xtol/2``, as ``brentq`` provides.
        """

        dense_output = True
        if np.isscalar(tspan):
            t0 = 0
            tF = tspan
        elif len(tspan) == 2:
            t0 = tspan[0]
            tF = tspan[1]
        else:
            dense_output = False
            t0 = tspan[0]
            tF = tspan[-1]

        if dense_output:
            tspan = np.array([t0, tF])
        else:
            tspan = np.array(tspan)

        """
        tspan is used to indicate which times must be computed
        these are end-points for continuous time simulations, meshed data 
        points for continuous.

        """

        if ('max_step' in integrator_options) and \
             (integrator_options['max_step'] == 0.0):
            integrator_options = integrator_options.copy()
            # TODO: find the harmonic to ensure no skipped steps?
            if np.any(self.dts!=0.0):
                integrator_options['max_step'] = np.min(self.dts[self.dts!=0.0])/2



        # generate tresult arrays; initialize x0
        results = SimulationResult(self.cum_states[-1], self.cum_outputs[-1],
                                   tspan, self.systems.size)

        def continuous_time_integration_step(t, state, for_integrator=True):
            """
            function to manipulate stored states and integrator state
            to pass to between computation_step and integrator
            """
            comp_result = self.computation_step(
                  t, state.reshape(-1), selector=True, do_events=~for_integrator)
            if not for_integrator:
                return (state,) + comp_result[1:]
            return comp_result[0]

        # store the results from each continuous integration step
        def collect_integrator_results(t, state):
            dxdt, output, events = \
                continuous_time_integration_step(t, state,
                    for_integrator=False)
            test_sel = results.res_idx - np.arange(3)-1
            if (t in results.t[test_sel] and
                    state in results.x[test_sel, :] and
                    output in results.y[test_sel, :]):
                return

            # check for events here -- before saving, because it is potentially
            # invalid
            prev_events = results.e[results.res_idx-1, :]
            if (np.any(np.sign(prev_events) != np.sign(events)) &
                (results.t[results.res_idx-1] > 0)):
                return -1
            else:
                results.new_result(t, state, output, events)

            if np.any(np.isnan(output)):
                warnings.warn(nan_warning_message.format(
                        "variable step-size collection",
                        t,
                        state,
                        output
                    ))
                return -1

        for sys in self.systems:
            sys.prepare_to_integrate()
        x0 = self.initial_condition

        # initial condition computation, populate initial condition in results

        #
        # Initial event computation
        #

        # compute first output for stateful systems
        y0 = self.output_equation_function(t0, x0, update_memoryless_event=True)

        dx_dt_0, y0, e0 = self.computation_step( # TODO: this is where logic for events needs to happen
             t0, x0, y0, selector=True, do_events=True)
        # initial_computation[0] is saved for the next round of selected DTs
        results.new_result(t0, x0, y0, e0)
        prev_event_t = t0

        # setup the integrator
        r = integrator_class(continuous_time_integration_step)
        r.set_integrator(**integrator_options)
        r.set_initial_value(x0, t0)
        if dense_output:
            r.set_solout(collect_integrator_results)

        # main simulation loop
        t_idx = 0
        next_t = tspan[1]
        # TODO: fix extra points being added to results
        while True:
            if np.any(np.isnan(results.y[:results.res_idx, :])):
                warnings.warn(nan_warning_message.format(
                        "tspan iteration (after event or meshed time-step)",
                        tspan[t_idx-1],
                        results.x[results.res_idx-1, :],
                        results.y[results.res_idx-1, :]
                    ))
                break

            # loop to integrate until next_t, while handling events
            try:
                r.integrate(next_t)
            except KeyboardInterrupt as kbi:
                break

            """
            possible branches:
                1. if dense:
                    a. event occured, process it                    
                    b. integration completed (to next_t), so exit
                    c. some other error, abort

                2. if meshed:
                    a. event occured, process it
                    b. mesh point achieved, no event
                        i. if next_t == tF, exit
                        ii. otherwise, do the next one.
                    c. some other error, abort

                1b, 2b, require adding the final point to the system (maybe not 1b)
                1a and 2a are the same, except if not dense, maybe don't save the point?? mesh should have fixed output datasize
                or, just don't allow meshed datapoints??
                1c and 2c are the same

                TODO: decide what to do about meshed data points, stiff solvers
                TODO: figure out how to run tests that don't involve those solvers
            """

            if dense_output:
                latest_t, latest_states, latest_outputs = \
                    results.last_result()
                if r.t == next_t or np.any(np.isnan(latest_outputs)):
                    break

            check_states, check_outputs, check_events = \
                continuous_time_integration_step(r.t, r.y, False)

            if np.any(np.isnan(check_outputs)):
                warnings.warn(nan_warning_message.format(
                        "tspan iteration after continuous integration",
                        r.t,
                        check_states,
                        check_outputs
                    ))
                break

            if (not dense_output and
                    np.all(
                        np.sign(results.e[results.res_idx-1, :]) ==
                        np.sign(check_events)
                    )):
                latest_states, latest_outputs, = \
                    check_states, check_outputs
                break

            if not r.successful():
                warnings.warn("Integrator quit unsuccessfully.")
                break

            #
            # need to handle event
            #

            # results index from previous event crossing
            prev_event_idx = np.where(
                results.t[:results.res_idx, None] == prev_event_t
             )[0][-1]
            prev_event_idx = max(
                min(prev_event_idx, results.res_idx-3), 0
            )

            # find which system(s) crossed
            event_cross_check = (
                np.sign(results.e[results.res_idx-1, :]) !=
                np.sign(check_events)
            )
            event_index_crossed = np.where(event_cross_check)[0]

            # interpolate to find first t crossing
            # holds t's where event occured
            event_ts = np.zeros(self.systems.size)
            # holds callable for root finding
            event_searchables = np.empty(self.systems.size,
                                         dtype=object)
            event_callables = np.empty(self.systems.size,
                                       dtype=object)

            ts_to_collect = np.r_[
                results.t[prev_event_idx:results.res_idx],
            ]

            unique_ts_to_collect, unique_ts_to_collect_idx = \
                np.unique(ts_to_collect, return_index=True)

            #

            # use vars check_states, check_outputs, check_events, r.t, (r.y?)
            # in interpolatant
            PRE_CROSS_MINIMUM = 3 # interpolant requires 4, I think, so 3 before the crossing

            crossed_size = max(
                PRE_CROSS_MINIMUM - unique_ts_to_collect.size, 0
                ) + 1
            crossed_times = np.zeros(crossed_size)
            crossed_states = np.zeros((crossed_size, self.cum_states[-1]))
            crossed_outputs = np.zeros((crossed_size, self.cum_outputs[-1]))
            crossed_events = np.zeros((crossed_size, self.systems.size))
            # use array allow in scope of result collector; not sure if needed
            crossed_idx = [0]
            
            def collect_integrator_results_events(t, state):
                dxdt, output, events = \
                    continuous_time_integration_step(t, state,
                        for_integrator=False)

                test_sel = results.res_idx - np.arange(3)-1
                if ((t in results.t[test_sel] and
                        state in results.x[test_sel, :] and
                        output in results.y[test_sel, :]) or 
                    (t in crossed_times and state in crossed_states and
                        output in crossed_outputs)):
                    return

                crossed_times[crossed_idx[0]] = t
                crossed_states[crossed_idx[0],:] = state
                crossed_outputs[crossed_idx[0],:] = output
                crossed_events[crossed_idx[0],:] = events
                crossed_idx[0] += 1

                if (crossed_idx[0] >= crossed_size):
                    return -1


            r.set_initial_value(r.y, r.t)
            r.set_solout(collect_integrator_results_events)
            r.integrate(next_t)

            if dense_output:
                r.set_solout(collect_integrator_results)
            else:
                r.set_solout(lambda *args: None)
            ts_for_interpolant = np.r_[unique_ts_to_collect, crossed_times]

            state_values = results.x[
                prev_event_idx:results.res_idx,
            ]
            state_values = np.r_[
                state_values[unique_ts_to_collect_idx, :],
                crossed_states
            ]

            state_traj_callable = callable_from_trajectory(
                ts_for_interpolant,
                state_values
            )

            output_values = results.y[prev_event_idx:results.res_idx]

            output_values = np.r_[
                output_values[unique_ts_to_collect_idx, :],
                crossed_outputs
            ]

            output_traj_callable = callable_from_trajectory(
                ts_for_interpolant,
                output_values
            )

            for sysidx in event_index_crossed:
                sys = self.systems[sysidx]

                state_start = self.cum_states[sysidx]
                state_end = self.cum_states[sysidx+1]

                input_start = self.cum_inputs[sysidx]
                input_end = self.cum_inputs[sysidx+1]

                if sys.dim_state:
                    event_searchables[sysidx] = \
                        lambda t: sys.event_equation_function(
                            t, state_traj_callable(t)[..., 
                                state_start:state_end]
                        )
                else:
                    event_searchables[sysidx] = \
                        lambda t: sys.event_equation_function(
                            t, output_traj_callable(t)[..., 
                                np.where(
                                    self.connections[
                                        :, input_start:input_end
                                    ].T
                                )[1]
                            ]
                        )
                if np.prod(np.sign(np.r_[
                  event_searchables[sysidx](results.t[prev_event_idx]),
                  event_searchables[sysidx](r.t)])) not in [0,-1]:
                        e_checks = np.r_[
                            results.e[
                                prev_event_idx:results.res_idx,
                                sysidx
                            ],
                            check_events[sysidx]
                        ]
                        left_bracket_idx = np.where(
                            np.sign(e_checks[:-1]) !=
                            np.sign(e_checks[-1])
                        )[0][-1]
                        left_bracket = ts_to_collect[left_bracket_idx]
                else:
                    left_bracket = results.t[prev_event_idx]
                    event_ts[sysidx] = event_finder(
                        event_searchables[sysidx],
                        left_bracket + np.finfo(np.float_).eps,
                        r.t,
                        **event_find_options
                    )

            next_event_t = np.min(event_ts[event_index_crossed])
            left_t = next_event_t-event_find_options['xtol']/2
            left_x = state_traj_callable(left_t)

            new_states, new_outputs, new_events = \
                continuous_time_integration_step(
                    left_t, left_x, False)
            results.new_result(
                left_t, new_states, new_outputs, new_events)

            right_t = next_event_t+event_find_options['xtol']/2
            right_x = state_traj_callable(right_t).reshape(-1)
            right_y = output_traj_callable(right_t).reshape(-1)

            # need to update the output for any stateful, probably do full pattern (i.e., output of system with state and event, output of system (with event only?), etc. Or just leave like this ssince it works??) from continuous_time_integration_step (for_integrator = False)
            # TODO: when cleaning up the integration loops, clean the event update too! 
            update_equation_function_indexes = np.where(
                event_cross_check & (event_ts == next_event_t)
            )[0]

            for sysidx in update_equation_function_indexes:
                sys = self.systems[sysidx]
                output_start = self.cum_outputs[sysidx]
                output_end = self.cum_outputs[sysidx+1]
                input_start = self.cum_inputs[sysidx]
                input_end = self.cum_inputs[sysidx+1]
                input_values = right_y[np.where(
                    self.connections[:, input_start:input_end].T
                )[1]]
                state_start = self.cum_states[sysidx]
                state_end = self.cum_states[sysidx+1]
                state_values = right_x[state_start:state_end]
                if sys.dim_state and sys.dim_input:
                    update_return_value = sys.update_equation_function(
                      right_t,
                      state_values,
                      input_values
                    )
                elif sys.dim_state:
                    update_return_value = sys.update_equation_function(
                      right_t,
                      state_values
                    )
                elif sys.dim_input:
                    update_return_value = sys.update_equation_function(
                      right_t, input_values)
                else:
                    update_return_value = sys.update_equation_function(
                      right_t)
                if sys.dim_state:
                    right_x[state_start:state_end] = \
                        update_return_value.reshape(-1)

                    right_y[output_start:output_end] = \
                        sys.output_equation_function(right_t, update_return_value).squeeze()
                elif sys.dim_input:
                    right_y[output_start:output_end] = \
                        sys.output_equation_function(right_t, input_values).squeeze()
                else:
                    right_y[output_start:output_end] = \
                        sys.output_equation_function(right_t).squeeze()

            new_states, new_outputs, new_events = \
                continuous_time_integration_step(
                    right_t, right_x, False)
            results.new_result(
                right_t, new_states, new_outputs, new_events)

            # set x (r.y), store in result as t+epsilon? if not dense,
            # add extra 1=-0
            r.set_initial_value(right_x, right_t)
            prev_event_t = right_t
            # TODO: THIS IS WHERE PREVIOUS EVENT HANDLING LOOP ENDED

        results.t = results.t[:results.res_idx]
        results.x = results.x[:results.res_idx, :]
        results.y = results.y[:results.res_idx, :]
        results.e = results.e[:results.res_idx, :]
        return results
