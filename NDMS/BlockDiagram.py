from scipy.integrate import ode
import numpy as np
import types
from .utils import process_vector_args

# TODO: use pandas for the results (symbol column index)
# TODO: create custom dataframe that automatically computes column expressions if all atoms are present
# TODO: make it so strings get sympified to try to find columns?
class SimulationResult(object):
    def __init__(self, n_states, n_outputs):
        self.t = np.array([])
        self.x = np.array([]).reshape([0,n_states])
        self.y = np.array([]).reshape([0,n_outputs])

class BlockDiagram(object):
    def __init__(self, *systems):
        if len(systems) == 0:
            self.systems = np.array([], dtype=object)
            self.connections = np.array([],dtype=np.int_).reshape((0,0)) # or bool??
            self.dts = np.array([],dtype=np.float_)
            self.cum_inputs = np.array([0],dtype=np.int_)
            self.cum_outputs = np.array([0],dtype=np.int_)
            self.cum_states = np.array([0],dtype=np.int_)
        else:
            self.systems = np.array(systems, dtype=object)

            self.dts = np.zeros_like(self.systems,dtype=np.float_)
            self.cum_inputs = np.zeros(self.systems.size+1,dtype=np.int_)
            self.cum_outputs = np.zeros(self.systems.size+1,dtype=np.int_)
            self.cum_states = np.zeros(self.systems.size+1,dtype=np.int_)

            for i,sys in enumerate(self.systems):
                self.dts[i] = sys.dt
                self.cum_inputs[i+1] = self.cum_inputs[i]+sys.n_inputs
                self.cum_outputs[i+1] = self.cum_outputs[i]+sys.n_outputs
                self.cum_states[i+1] = self.cum_states[i]+sys.n_states

            self.connections = np.zeros((self.cum_outputs[-1],self.cum_inputs[-1]),dtype=np.bool_)

    def connect(self, from_system_output, to_system_input, outputs=[], inputs=[]):
        if outputs==[]:
            outputs = np.arange(from_system_output.n_outputs)
        else:
            outputs = np.asarray(outputs)
        outputs = outputs+self.cum_outputs[np.where(self.systems==from_system_output)]

        if inputs==[]:
            inputs = np.arange(to_system_input.n_inputs)
        else:
            inputs = np.asarray(inputs)
        inputs = inputs+self.cum_inputs[np.where(self.systems==to_system_input)]

        self.connections[:,inputs] = False # reset old connections
        self.connections[outputs,inputs] = True

    def add_system(self, system):
        self.systems = np.append(self.systems, system)
        self.cum_states = np.append(self.cum_states,self.cum_states[-1]+system.n_states)
        self.cum_inputs = np.append(self.cum_inputs,self.cum_inputs[-1]+system.n_inputs)
        self.cum_outputs = np.append(self.cum_outputs,self.cum_outputs[-1]+system.n_outputs)
        self.dts = np.append(self.dts,system.dt)
        self.connections = np.pad(self.connections,((0,system.n_outputs),(0,system.n_inputs)),'constant',constant_values=0)

    def simulate(self, tspan, integrator_name='dopri5', integrator_options={}):
        """
        TODO: recreate into giant expression, hopefully with CSE? This would
        speed up CT systems, but most likely the interesting ones are algorithmic.

        So goal of symbolic manipulation is to create efficient callable
        that is only a function of states and possibly time. Then efficient 
        (i.e., vectorized?) to generate the outputs of each system which may not
        get stored anymore.

        Related: do we ever benefit from implicit form? explicit is probably
        always fine numerically? With DT and especially algorithmic, these
        inefficiencies may be neglible. 

        Could also group symbolic discrete time systems by dt. But ultimately 
        this still needs to handle "algorithmic" systems (controllers?) actually,
        if hybrid then do it by DTs and know integrator will stop at a
        good time.

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

        if hybrid:
            tspan = np.array([])
            for dt in np.unique(self.dts):
                if dt:
                    tspan = np.unique(np.append(tspan,np.arange(t0,tF+dt,dt)))
        elif np.isscalar(tspan):
            tspan = np.array([t0, tF])
        """
        tspan is used to indicate which times must be computed
        these are the time-steps for a discrete simulation, end-points for
        continuous time simulations, meshed data points for continuous
        time simulations, and times where discrete systems/controllers fire
        for (time) hybrid systems
        """


        # generate tresult arrays; initialize x0
        results = SimulationResult(self.cum_states[-1],self.cum_outputs[-1])
        all_x0 = np.array([])
        ct_x0 = np.array([])
        for sys in self.systems:
            sys.prepare_to_integrate()
            all_x0 = np.append(all_x0,sys.initial_condition)
            if sys.dt == 0:
                ct_x0 = np.append(ct_x0,sys.initial_condition)

        def computation_step(t,states_in,outputs_in,selector=None):
            """
            callable to compute system outputs and state derivatives
            """
            states = np.copy(states_in)
            outputs = np.copy(outputs_in)

            # compute outputs for full systems, y[t_k]=g(t_k,x[t_k])
            for sysidx in np.where((np.diff(self.cum_states)>0)&selector)[0]:
                sys = self.systems[sysidx]
                output_start = self.cum_outputs[sysidx]
                output_end = self.cum_outputs[sysidx+1]
                state_start = self.cum_states[sysidx]
                state_end = self.cum_states[sysidx+1]

                state_values = states[state_start:state_end]
                outputs[output_start:output_end] = sys.output_equation_function(t,state_values).reshape(-1)

            # compute outputs for memoryless systems, y[t_k]=g(t_k,u[t_k])
            for sysidx in np.where((np.diff(self.cum_states)==0)&selector)[0]:
                sys = self.systems[sysidx]
                output_start = self.cum_outputs[sysidx]
                output_end = self.cum_outputs[sysidx+1]
                input_start = self.cum_inputs[sysidx]
                input_end = self.cum_inputs[sysidx+1]
                input_where_x,input_where_y = np.where(self.connections[:,input_start:input_end])
                input_values = outputs[input_where_x[input_where_y]]
                if len(input_values):
                    outputs[output_start:output_end] = sys.output_equation_function(t,input_values).reshape(-1)
                else:
                    outputs[output_start:output_end] = sys.output_equation_function(t).reshape(-1)

            # compute outputs for full systems, x[t_k+dt]=f(t_k,x[t_k],u[t_k])
            for sysidx in np.where((np.diff(self.cum_states)>0)&selector)[0]:
                sys = self.systems[sysidx]

                state_start = self.cum_states[sysidx]
                state_end = self.cum_states[sysidx+1]
                state_values = states[state_start:state_end]

                input_start = self.cum_inputs[sysidx]
                input_end = self.cum_inputs[sysidx+1]
                input_where_x,input_where_y = np.where(self.connections[:,input_start:input_end])
                input_values = outputs[input_where_x[input_where_y]]

                states[state_start:state_end] = sys.state_equation_function(t,state_values,input_values).reshape(-1)

            return states,outputs

        def continuous_time_integration_step(t,ct_states,for_integrator=True):
            """
            function to manipulate stored states and integrator state
            to pass to between computation_step and integrator
            """
            ct_selector = (self.dts==0)
            outputs = np.copy(results.y[-1,:])
            states = np.copy(results.x[-1,:])

            # pass the integrator's current values to the computation step
            ct_state_accumulator = 0
            for sysidx in np.where((np.diff(self.cum_states)>0)&ct_selector)[0]:
                sys = self.systems[sysidx]
                state_start = self.cum_states[sysidx]
                state_end = self.cum_states[sysidx+1]
                states[state_start:state_end] = ct_states[ct_state_accumulator:ct_state_accumulator+sys.n_states]
                ct_state_accumulator += sys.n_states

            comp_states,comp_out = computation_step(t,states,outputs,selector=ct_selector)

            if not for_integrator: # i.e., to collect the results
                return comp_states,comp_out

            # return the comptued derivatives to the integrator
            ct_derivative = np.zeros_like(ct_states)
            ct_state_accumulator = 0
            for sysidx in np.where((np.diff(self.cum_states)>0)&ct_selector)[0]:
                sys = self.systems[sysidx]
                state_start = self.cum_states[sysidx]
                state_end = self.cum_states[sysidx+1]
                ct_derivative[ct_state_accumulator:ct_state_accumulator+sys.n_states] = comp_states[state_start:state_end]
                ct_state_accumulator += sys.n_states

            return ct_derivative

        # store the results from each continuous integration step
        def collect_integrator_results(t,ct_states):
            new_states,new_outputs = continuous_time_integration_step(t,ct_states,False)
            results.t = np.append(results.t, t)
            results.x = np.vstack((results.x, new_states.reshape((1,-1))))
            results.y = np.vstack((results.y, new_outputs.reshape((1,-1))))

            if np.any(np.isnan(new_outputs)):
                print("aborting")
                return -1

        # TODO: decouple integrator; perhaps use PyDy, PyODEsys, PyDStool, etc
        # setup the integrator if we have CT states
        if len(ct_x0) > 0:
            r = ode(continuous_time_integration_step)
            r.set_integrator(integrator_name, **integrator_options)
            r.set_initial_value(ct_x0,t0)
            if dense_output:
                r.set_solout(collect_integrator_results)

        # initial condition computation, populate initial condition in results
        # I am not sure if the timing is right for DT memoryless systems; they may need to be shifted

        """compute the outputs based on initial conditions; add initial conditions
        and initial outputs to results"""
        y0_zeros = np.zeros(self.cum_outputs[-1])
        initial_computation = computation_step(t0,all_x0,y0_zeros,np.ones_like(self.systems))
        results.t = np.append(results.t, t0)
        results.x = np.vstack((results.x,all_x0.reshape((1,-1))))
        results.y = np.vstack((results.y, initial_computation[1].reshape((1,-1))))
        for next_t in tspan[1:]:
            if len(ct_x0) > 0:
                r.set_initial_value(r.y,r.t)
                r.integrate(next_t)
                if not dense_output:
                    latest_states, latest_outputs = continuous_time_integration_step(r.t,r.y,False)
                else:
                    latest_states = results.x[-1,:] 
                    latest_outputs = results.y[-1,:]

                dt_time_selector = (np.mod(next_t,self.dts)==0)
                if np.any(np.isnan(results.y)):
                    break
            else:
                latest_states = results.x[-1,:] # inherently not dense
                latest_outputs = results.y[-1,:]
                dt_time_selector = ((np.mod(next_t,self.dts)==0)|(self.dts==0))
            new_states,new_outputs = computation_step(next_t,latest_states,latest_outputs,dt_time_selector)
            results.t = np.append(results.t, next_t)
            results.x = np.vstack((results.x, new_states.reshape((1,-1))))
            results.y = np.vstack((results.y, new_outputs.reshape((1,-1))))

        return results


