from scipy.integrate import ode
import numpy as np
import warnings
from simupy.utils import callable_from_trajectory
from scipy.optimize import brentq

DEFAULT_INTEGRATOR_NAME = 'dopri5'

DEFAULT_INTEGRATOR_OPTIONS = {
        'rtol': 1e-6,
        'atol': 1e-12,
        'nsteps': 500,
        'max_step': 0.0
    }

DEFAULT_EVENT_FIND_OPTIONS = {
        'xtol': 2e-12,
        'rtol': 8.8817841970012523e-16,
        'maxiter': 100
    }

nan_warning_message = ("BlockDiagram encountered NaN outputs and quit during " +
    " {}. This may have been intentional! NaN outputs at occured index {} " +
    " at time t={} and state x={}")


class SimulationResult(object):
    # TODO: use pandas for the results (symbol column index)
    # TODO: create custom dataframe that automatically computes column
    # expressions if all atoms are present
    # TODO: make it so strings get sympified to try to find columns?
    # TODO: enforce naming so things don't clash? and then a mechanism to
    # simplify names? and naming schemes for non DynamicalSystem systems.

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
        more_rows = min(more_rows, self.max_allocation)

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
    def __init__(self, *systems):
        if len(systems) == 0:
            self.systems = np.array([], dtype=object)
            self.connections = np.array([], dtype=np.bool_).reshape((0, 0))
            self.dts = np.array([], dtype=np.float_)
            self.events = np.array([], dtype=np.bool_)
            self.cum_inputs = np.array([0], dtype=np.int_)
            self.cum_outputs = np.array([0], dtype=np.int_)
            self.cum_states = np.array([0], dtype=np.int_)
            self.cum_events = np.array([0], dtype=np.int_)
        else:
            self.systems = np.array(systems, dtype=object)

            self.dts = np.zeros_like(self.systems, dtype=np.float_)
            self.events = np.zeros_like(self.systems, dtype=np.bool_)
            self.cum_inputs = np.zeros(self.systems.size+1, dtype=np.int_)
            self.cum_outputs = np.zeros(self.systems.size+1, dtype=np.int_)
            self.cum_states = np.zeros(self.systems.size+1, dtype=np.int_)
            self.cum_events = np.zeros(self.systems.size+1, dtype=np.int_)

            for i, sys in enumerate(self.systems):
                self.dts[i] = sys.dt
                self.events[i] = (hasattr(sys, 'event_equation_function') and
                                  hasattr(sys, 'update_equation_function'))
                self.cum_inputs[i+1] = self.cum_inputs[i] + sys.dim_input
                self.cum_outputs[i+1] = self.cum_outputs[i] + sys.dim_output
                self.cum_states[i+1] = self.cum_states[i] + sys.dim_state
                self.cum_events[i+1] = self.cum_events[i] + self.events[i]

            self.connections = np.zeros(
                    (self.cum_outputs[-1], self.cum_inputs[-1]),
                    dtype=np.bool_)

    def connect(self, from_system_output, to_system_input, outputs=[],
                inputs=[]):
        if outputs == []:
            outputs = np.arange(from_system_output.dim_output)
        else:
            outputs = np.asarray(outputs)
        outputs = outputs + self.cum_outputs[
                        np.where(self.systems == from_system_output)
                    ]

        if inputs == []:
            inputs = np.arange(to_system_input.dim_input)
        else:
            inputs = np.asarray(inputs)
        inputs = inputs + self.cum_inputs[
                      np.where(self.systems == to_system_input)
                  ]

        self.connections[:, inputs] = False  # reset old connections
        self.connections[outputs, inputs] = True

    def add_system(self, system):
        self.systems = np.append(self.systems, system)
        self.cum_states = np.append(self.cum_states,
                                    self.cum_states[-1] + system.dim_state)
        self.cum_inputs = np.append(self.cum_inputs,
                                    self.cum_inputs[-1] + system.dim_input)
        self.cum_outputs = np.append(self.cum_outputs,
                                     self.cum_outputs[-1] + system.dim_output)
        self.cum_outputs = np.append(self.cum_outputs,
                                     self.cum_outputs[-1] + system.dim_output)
        self.events = np.append(self.events,
                                (hasattr(system, 'event_equation_function') and
                                 hasattr(system, 'update_equation_function')))
        self.cum_events = np.append(self.cum_events,
                                    self.cum_events[-1] + self.events[-1])
        self.dts = np.append(self.dts, system.dt)
        self.connections = np.pad(self.connections,
                                  ((0, system.dim_output),
                                   (0, system.dim_input)),
                                  'constant', constant_values=0)

    def simulate(self, tspan, integrator_name=DEFAULT_INTEGRATOR_NAME,
                 integrator_options=DEFAULT_INTEGRATOR_OPTIONS,
                 event_find_options=DEFAULT_EVENT_FIND_OPTIONS):
        """
        TODO: recreate into giant expression, hopefully with CSE? This would
        speed up CT systems, but most likely the interesting ones are hybrid.

        So goal of symbolic manipulation is to create efficient callable
        that is only a function of states and possibly time. Then efficient
        (i.e., vectorized?) to generate the outputs of each system which may
        not get stored anymore.

        Related: do we ever benefit from implicit form? explicit is probably
        always fine numerically? With DT and especially algorithmic, these
        inefficiencies may be neglible.

        Some will require a jacobian, not sure how inaccurate or inefficient it
        would be just to follow the same computation procedure used here

        Could also group symbolic discrete time systems by dt. But ultimately
        this still needs to handle "algorithmic" systems (controllers?)
        actually, if hybrid then do it by DTs and know integrator will stop at
        a good time.

        also need to decouple code generation from this and from System
        """

        # generate tspan based on DTs, add hybrid flag
        if len(np.unique(self.dts)) > 1:
            hybrid = True
        else:
            hybrid = False

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

        all_dt_sel = np.zeros((tspan.size, self.dts.size), dtype=np.bool)
        if hybrid or 0 not in self.dts:
            all_dts = [np.arange(t0, tF+dt, dt) if dt != 0 else np.r_[t0, tF]
                       for dt in self.dts]
            tspan = np.unique(np.concatenate([tspan]+all_dts))

            for idx, dt in enumerate(self.dts):
                if dt != 0:
                    all_dt_sel[:, idx] = np.any(np.equal(
                            *np.meshgrid(all_dts[idx], tspan)), axis=1)

        """
        tspan is used to indicate which times must be computed
        these are the time-steps for a discrete simulation, end-points for
        continuous time simulations, meshed data points for continuous
        time simulations, and times where discrete systems/controllers fire
        for (time) hybrid systems

        all_dts is used to select which dt systems should compute at the time
        step. TODO: is there a faster way?
        """
        # generate tresult arrays; initialize x0
        results = SimulationResult(self.cum_states[-1], self.cum_outputs[-1],
                                   tspan, self.systems.size)

        all_x0 = np.array([])  # TODO: pre-allocate?
        ct_x0 = np.array([])
        for sys in self.systems:
            sys.prepare_to_integrate()
            all_x0 = np.append(all_x0, sys.initial_condition)
            if sys.dt == 0:
                ct_x0 = np.append(ct_x0, sys.initial_condition)

        state_sel_ct = np.empty(0, dtype=np.int_)
        for sysidx in np.where(
                (np.diff(self.cum_states) > 0) & (self.dts == 0))[0]:
            sys = self.systems[sysidx]
            state_start = self.cum_states[sysidx]
            state_end = self.cum_states[sysidx+1]
            state_sel_ct = np.r_[state_sel_ct,
                                 np.arange(state_start, state_end)]

        def computation_step(t, states_in, outputs_in, selector=None,
                             do_events=False):
            """
            callable to compute system outputs and state derivatives
            """
            states = np.copy(states_in)
            outputs = np.copy(outputs_in)

            # compute outputs for full systems, y[t_k]=h(t_k,x[t_k])
            # TODO: Is it possible to refactor these loops using a function
            # that takes the total selector, input name, output name, and
            # callable name?
            for sysidx in np.where(
                    (np.diff(self.cum_states) > 0) & selector)[0]:
                sys = self.systems[sysidx]
                output_start = self.cum_outputs[sysidx]
                output_end = self.cum_outputs[sysidx+1]
                state_start = self.cum_states[sysidx]
                state_end = self.cum_states[sysidx+1]

                state_values = states[state_start:state_end]
                outputs[output_start:output_end] = \
                    sys.output_equation_function(t, state_values).reshape(-1)

            # compute outputs for memoryless systems, y[t_k]=h(t_k,u[t_k])
            for sysidx in np.where(
                    (np.diff(self.cum_states) == 0) & selector)[0]:
                sys = self.systems[sysidx]
                output_start = self.cum_outputs[sysidx]
                output_end = self.cum_outputs[sysidx+1]
                input_start = self.cum_inputs[sysidx]
                input_end = self.cum_inputs[sysidx+1]
                input_values = outputs[
                        np.where(
                            self.connections[:, input_start:input_end].T
                        )[1]]
                if len(input_values):
                    outputs[output_start:output_end] = \
                      sys.output_equation_function(t, input_values).reshape(-1)
                else:
                    outputs[output_start:output_end] = \
                      sys.output_equation_function(t).reshape(-1)

            # compute state equation for full systems,
            # x[t_k']=f(t_k,x[t_k],u[t_k])
            for sysidx in np.where(
                    (np.diff(self.cum_states) > 0) & selector)[0]:
                sys = self.systems[sysidx]

                state_start = self.cum_states[sysidx]
                state_end = self.cum_states[sysidx+1]
                state_values = states[state_start:state_end]

                input_start = self.cum_inputs[sysidx]
                input_end = self.cum_inputs[sysidx+1]
                input_values = outputs[
                        np.where(
                                self.connections[:, input_start:input_end].T
                        )[1]]

                states[state_start:state_end] = \
                    sys.state_equation_function(
                        t, state_values, input_values).reshape(-1)

            if do_events:
                events = np.zeros(self.systems.size)

                for sysidx in np.where(
                            (np.diff(self.cum_states) > 0) & self.events &
                            selector
                        )[0]:
                    sys = self.systems[sysidx]
                    state_start = self.cum_states[sysidx]
                    state_end = self.cum_states[sysidx+1]
                    state_values = states[state_start:state_end]
                    events[sysidx] = sys.event_equation_function(
                                        t, state_values).reshape(-1)

                # compute outputs for memoryless systems, y[t_k]=h(t_k,u[t_k])
                for sysidx in np.where(
                            (np.diff(self.cum_states) == 0) & self.events &
                            selector
                        )[0]:
                    sys = self.systems[sysidx]
                    input_start = self.cum_inputs[sysidx]
                    input_end = self.cum_inputs[sysidx+1]
                    input_values = outputs[
                            np.where(
                                self.connections[:, input_start:input_end].T
                            )[1]]
                    if len(input_values):
                        events[sysidx] = sys.event_equation_function(
                                                t, input_values).reshape(-1)
                    else:
                        events[sysidx] = sys.event_equation_function(
                                                t).reshape(-1)

                return states, outputs, events

            return states, outputs

        def continuous_time_integration_step(t, ct_states,
                                             for_integrator=True):
            """
            function to manipulate stored states and integrator state
            to pass to between computation_step and integrator
            """
            ct_selector = (self.dts == 0)
            prevt, states, outputs = results.last_result(copy=True)

            # pass the integrator's current values to the computation step
            # TODO: is there a more efficient way to do this? See how events
            # get selected
            ct_state_accumulator = 0
            for sysidx in np.where(
                    (np.diff(self.cum_states) > 0) & ct_selector)[0]:
                sys = self.systems[sysidx]
                state_start = self.cum_states[sysidx]
                state_end = self.cum_states[sysidx+1]
                states[state_start:state_end] = ct_states[
                        ct_state_accumulator:ct_state_accumulator+sys.dim_state
                    ]
                ct_state_accumulator += sys.dim_state

            if not for_integrator:  # i.e., to collect the results
                comp_states, comp_out, comp_events = computation_step(
                      t, states, outputs, selector=ct_selector, do_events=True)
                return states, comp_out, comp_events

            comp_states, comp_out = computation_step(
                    t, states, outputs, selector=ct_selector)

            # return the comptued derivatives to the integrator
            ct_derivative = np.zeros_like(ct_states)
            ct_state_accumulator = 0
            for sysidx in np.where(
                    (np.diff(self.cum_states) > 0) & ct_selector)[0]:
                sys = self.systems[sysidx]
                state_start = self.cum_states[sysidx]
                state_end = self.cum_states[sysidx+1]
                ct_derivative[
                        ct_state_accumulator:ct_state_accumulator+sys.dim_state
                    ] = comp_states[state_start:state_end]
                ct_state_accumulator += sys.dim_state

            return ct_derivative

        # store the results from each continuous integration step
        def collect_integrator_results(t, ct_states):
            new_states, new_outputs, new_events = \
                continuous_time_integration_step(t, ct_states, False)
            test_sel = results.res_idx - np.arange(3)-1
            if (t in results.t[test_sel] and
                    new_states in results.x[test_sel, :] and
                    new_outputs in results.y[test_sel, :]):

                return

            # check for events here -- before saving, because it is potentially
            # invalid
            if np.any(
                        np.sign(results.e[results.res_idx-1, :]) !=
                        np.sign(new_events)
                    ):
                return -1
            else:
                results.new_result(t, new_states, new_outputs, new_events)

            if np.any(np.isnan(new_outputs)):
                np.where(np.isnan(new_outputs))
                warnings.warn(nan_warning_message.format({
                        "variable step-size collection",
                        str(np.where(np.isnan(new_outputs))[0]),
                        t,
                        new_states
                    }))
                return -1

        # TODO: decouple integrator; perhaps use PyDy, PyODEsys, PyDStool,
        # Sundials, etc
        # setup the integrator if we have CT states
        if len(ct_x0) > 0:
            r = ode(continuous_time_integration_step)
            r.set_integrator(integrator_name, **integrator_options)
            r.set_initial_value(ct_x0, t0)
            if dense_output:
                r.set_solout(collect_integrator_results)

        # initial condition computation, populate initial condition in results
        # I am not sure if the timing is right for DT memoryless systems; they
        # may need to be shifted

        """compute the outputs based on initial conditions; add initial conditions
        and initial outputs to results"""
        y0 = np.zeros(self.cum_outputs[-1])

        # compute outputs for stateful systems that have no events:
        for sysidx in np.where(
                    (np.diff(self.cum_states) > 0) & (~self.events) &
                    (self.dts == 0)
                )[0]:
            sys = self.systems[sysidx]
            output_start = self.cum_outputs[sysidx]
            output_end = self.cum_outputs[sysidx+1]
            state_start = self.cum_states[sysidx]
            state_end = self.cum_states[sysidx+1]

            state_values = all_x0[state_start:state_end]
            y0[output_start:output_end] = sys.output_equation_function(
                    t0, state_values).reshape(-1)

        # compute outputs for memoryless systems that have no events
        for sysidx in np.where(
                    (np.diff(self.cum_states) == 0) & (~self.events) &
                    (self.dts == 0)
                )[0]:
            sys = self.systems[sysidx]
            output_start = self.cum_outputs[sysidx]
            output_end = self.cum_outputs[sysidx+1]
            input_start = self.cum_inputs[sysidx]
            input_end = self.cum_inputs[sysidx+1]
            input_values = y0[np.where(
                    self.connections[:, input_start:input_end].T)[1]]
            if len(input_values):
                y0[output_start:output_end] = sys.output_equation_function(
                        t0, input_values).reshape(-1)
            else:
                y0[output_start:output_end] = sys.output_equation_function(
                        t0).reshape(-1)

        # compute event update for stateful systems
        for sysidx in np.where(
                    (np.diff(self.cum_states) > 0) & self.events &
                    (self.dts == 0)
                )[0]:
            sys = self.systems[sysidx]
            state_start = self.cum_states[sysidx]
            state_end = self.cum_states[sysidx+1]
            state_values = all_x0[state_start:state_end]
            sys.update_equation_function(t0, state_values)

        # compute event update for memoryless systems, y[t_k]=h(t_k,u[t_k])
        for sysidx in np.where(
                    (np.diff(self.cum_states) == 0) & self.events &
                    (self.dts == 0)
                )[0]:
            sys = self.systems[sysidx]
            input_start = self.cum_inputs[sysidx]
            input_end = self.cum_inputs[sysidx+1]
            input_values = y0[np.where(
                    self.connections[:, input_start:input_end].T)[1]]
            if len(input_values):
                sys.update_equation_function(t0, input_values)
            else:
                sys.update_equation_function(t0)

        dt_time_selector = all_dt_sel[1, :]
        next_dt_x, y0, e0 = computation_step(
                t0, all_x0, y0, (dt_time_selector | (self.dts == 0)), True)
        # initial_computation[0] is saved for the next round of selected DTs
        results.new_result(t0, all_x0, y0, e0)
        prev_event_t = t0

        for t_idx, next_t in enumerate(tspan[1:]):
            if np.any(np.isnan(results.y[:results.res_idx, :])):
                break

            if len(ct_x0) > 0:  # handle continuous time integration
                r.set_initial_value(r.y, r.t)
                while True:
                    r.integrate(next_t)

                    if dense_output:
                        latest_t, latest_states, latest_outputs = \
                            results.last_result()
                        if r.t == next_t or np.any(np.isnan(latest_outputs)):
                            break

                    check_states, check_outputs, check_events = \
                        continuous_time_integration_step(r.t, r.y, False)

                    if np.any(np.isnan(check_outputs)):
                        warnings.warn(nan_warning_message.format({
                                "tspan iteration",
                                str(np.where(np.isnan(check_outputs))[0]),
                                r.t,
                                check_states
                            }))
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
                    event_index_crossed = np.where(
                        np.sign(results.e[results.res_idx-1, :]) !=
                        np.sign(check_events)
                    )[0]

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
                        r.t
                    ]

                    unique_ts_to_collect, unique_ts_to_collect_idx = \
                        np.unique(ts_to_collect, return_index=True)

                    for sysidx in event_index_crossed:
                        sys = self.systems[sysidx]

                        if sys.dim_state > 0:
                            input_start = self.cum_states[sysidx]
                            input_end = self.cum_states[sysidx+1]
                            input_values = results.x[
                                prev_event_idx:results.res_idx,
                                state_start:state_end
                            ]
                            input_values = np.r_[
                                input_values,
                                check_states[
                                    state_start:state_end].reshape(1, -1)
                            ]
                        else:
                            input_start = self.cum_inputs[sysidx]
                            input_end = self.cum_inputs[sysidx+1]
                            if input_end - input_start > 0:

                                input_values = results.y[
                                    prev_event_idx:results.res_idx,
                                    np.where(
                                        self.connections[
                                            :, input_start:input_end
                                        ].T
                                    )[1]
                                ]

                                input_values = np.r_[
                                    input_values,
                                    check_outputs[
                                        np.where(
                                            self.connections[
                                                :, input_start:input_end
                                            ].T
                                        )[1]
                                    ].reshape(1, -1)
                                ]

                            else:
                                input_values = results.t[
                                    prev_event_idx:results.res_idx
                                ]

                        input_traj_callable = callable_from_trajectory(
                            unique_ts_to_collect,
                            input_values[unique_ts_to_collect_idx, :]
                        )
                        event_callables[sysidx] = input_traj_callable
                        event_searchables[sysidx] = \
                            lambda t: sys.event_equation_function(
                                t, input_traj_callable(t)
                            )
                        if np.prod(np.sign(np.r_[
                          event_searchables[sysidx](results.t[prev_event_idx]),
                          event_searchables[sysidx](r.t)])) != -1:
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
                        event_ts[sysidx] = brentq(
                            event_searchables[sysidx],
                            left_bracket,
                            r.t,
                            **event_find_options
                        )

                    next_event_t = np.min(event_ts[event_index_crossed])
                    ct_state_traj_callable = callable_from_trajectory(
                        unique_ts_to_collect,
                        np.r_[
                            results.x[
                                prev_event_idx:results.res_idx,
                                state_sel_ct
                            ],
                            r.y.reshape(1, -1)
                        ][unique_ts_to_collect_idx, :]
                    )

                    left_t = next_event_t-event_find_options['xtol']/2
                    ct_xtleft = ct_state_traj_callable(left_t)

                    new_states, new_outputs, new_events = \
                        continuous_time_integration_step(
                            left_t, ct_xtleft, False)
                    results.new_result(
                        left_t, new_states, new_outputs, new_events)

                    right_t = next_event_t+event_find_options['xtol']/2
                    ct_xtright = ct_state_traj_callable(right_t)

                    for sysidx in np.where(event_ts == next_event_t)[0]:
                        sys = self.systems[sysidx]
                        update_return_value = sys.update_equation_function(
                            next_event_t,
                            event_callables[sysidx](next_event_t)
                        )
                        if sys.dim_state > 0:
                            ct_state_idx = np.where(
                                state_sel_ct == self.cum_states[sysidx]
                            )[0][0]
                            ct_xtright[
                                ct_state_idx:ct_state_idx+sys.dim_state+1
                            ] = update_return_value

                    new_states, new_outputs, new_events = \
                        continuous_time_integration_step(
                            right_t, ct_xtright, False)
                    results.new_result(
                        right_t, new_states, new_outputs, new_events)

                    # set x (r.y), store in result as t+epsilon? if not dense,
                    # add extra 1=-0
                    r.set_initial_value(ct_xtright, right_t)
                    prev_event_t = right_t
                prev_event_t = next_t
                dt_time_selector = all_dt_sel[t_idx+1, :]

            else:  # get previous computed states to begin discrete time
                # integration step
                latest_t, latest_states, latest_outputs = results.last_result()
                dt_time_selector = all_dt_sel[t_idx+1, :] | (self.dts == 0)

            # handle discrete integration steps
            latest_states = latest_states.copy()
            ct_time_slector = (
                (np.diff(self.cum_states) > 0) & all_dt_sel[t_idx+1, :]
            )
            for sysidx in np.where(ct_time_slector)[0]:
                sys = self.systems[sysidx]
                state_start = self.cum_states[sysidx]
                state_end = self.cum_states[sysidx+1]

                latest_states[state_start:state_end] = \
                    next_dt_x[state_start:state_end]

            next_states, current_outputs = computation_step(
                next_t, latest_states, latest_outputs, dt_time_selector)
            results.new_result(next_t, latest_states, current_outputs)
            if np.any(np.isnan(current_outputs)):
                break

            if next_t != tF:
                next_dt_x = next_states
        results.t = results.t[:results.res_idx]
        results.x = results.x[:results.res_idx, :]
        results.y = results.y[:results.res_idx, :]
        results.e = results.e[:results.res_idx, :]
        return results
