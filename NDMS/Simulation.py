from scipy.integrate import ode
import numpy as np
import types
from .utils import process_vector_args

class SimulationResult(object):
    def __init__(self, system, controller):
        self.t = np.empty(0)
        self.x = np.empty([0,system.n_states])
        self.u = np.empty([0,system.n_inputs])
        self.u_history = {}
        self.system = system
        self.controller = controller
        
        self.hybrid = False
        if hasattr(controller, 'dt') and not system.dt:
            self.hybrid = True
            
        # self.y_out = np.empty([0,n_sys_output])
    
    def get_key(self,t,x):
        if self.hybrid:
            # integrator must also do piecewise!
            key = process_vector_args((np.floor(t/self.controller.dt),))
        else:
            key = process_vector_args((t,x))
        
        return key
            

def IntegrateDiscreteTimeSystem(f, out, tspan, x0, dt):
    if np.isscalar(tspan):
        t0 = 0
        tF = tspan
    else:
        t0 = tspan[0]
        tF = tspan[-1]
    
    T = np.arange(t0, tF+dt, dt)
    # x = x0
    # f(0, x, out)
    # TODO: check math!
    """
    I believe it's
        x(k+1) = f(x(k),u(k))
        u(k+1) = f(t, x(k))
        
    no, it should be:
        x(k+1) = f(x(k),u(k))
        u(k) = g(t, x(k))
        
    so that
        x(k+1) = f(x(k), g(t, x(k))
    
    which is how the integration to call function should be working
    
    what about for augmented? now we're saying
    
    [ x(k+1).T, u(k).T ].T = f(x(k), u(k-1) + Du(k))
        
    """
    for t in T:
        if t == t0:
            x = x0
        
        out.t = np.append(out.t,[t],0)
        out.x = np.append(out.x,x.reshape((1,-1)),0)
        # out.u = np.zeros((1,)+(out.u.shape[1],)) # this line makes it so u(k) is shifted by 1 time interval

        x = f(t, x, out)
        if np.max(np.isnan(x)):
            break
        u = out.u_history[out.get_key(t,out.x[-1])]
        out.u = np.append(out.u,u,0)
    
    # TODO: Figure out how to filter dense output to meshed output
    
def IntegrateContinuousTimeSystem(f, out, tspan, x0):
    def collect_integrator_results(t, x, result):
        u = result.u_history[out.get_key(t,x)]
        # TODO: test that u is right shape for multi-output systems
        
        # TODO: check math!!
        """
        I think the system is
             dot x(t) = f(x(t), u(t))
             u(t) = controller(t, x(t))
             
        I believe this is how I am collecting the results
        """
        if np.max(np.isnan(x)):
            return -1
        if result.t.size == 0:
            result.t = np.array([t])
            result.x = x.reshape((1,)+x.shape)
            result.u = u.copy() # u.reshape((1,)+u.shape)
            # y_out = r.y.reshape(1,n_sys_output+1)
        else:
            result.t = np.append(result.t,t)
            result.x = np.append(result.x,x.reshape((1,)+x.shape),0)
            result.u = np.append(result.u,u,0)
            # result.u = np.append(result.u,u.reshape(1,system.n_inputs),0)
            # y_out = np.append(y_out,np.array(r.y).reshape(1,n_sys_output+1),0)
            
    dense_output = False
    meshed_output = False
    piecewise_output = False
    if np.isscalar(tspan):
        t0 = 0
        tF = tspan
        dense_output = True
    elif len(tspan) == 2:
        t0 = tspan[0]
        tF = tspan[1]
        dense_output = True
    else:
        num_time_points = len(tspan)
        t0 = tspan[0]
        tF = tspan[-1]
        meshed_output = True
    
    if out.hybrid:
        piecewise_output = True
        
    r = ode(f)
    r.set_integrator('dopri5')
    r.set_f_params(out)
    
    if dense_output:
        # from scipy\integrate\_ode.py, dopri5._solout
        # need to modify arguments to include an object to carry integration 
        # output data
        def new_dopri5_solout(self, nr, xold, x, y, nd, icomp, con):
            if self.solout is not None:
                if self.solout_cmplx:
                    y = y[::2] + 1j * y[1::2]
                # add object for generating output
                return self.solout(x, y, out)
            else:
                return 1
                
        r._integrator._solout = types.MethodType(new_dopri5_solout, r._integrator)
        r._integrator.call_args[2] = r._integrator._solout
        r.set_solout(collect_integrator_results)
        r.set_initial_value(x0, t0)
        if not piecewise_output:
            r.integrate(tF)
        else:
            original_tspan = tspan
            tspan = np.arange(t0, tF+out.controller.dt, out.controller.dt)
    if meshed_output or piecewise_output:
        for count_t, next_t in enumerate(tspan):
            if count_t == 0:
                r.set_initial_value(x0,t0)
            else:
                r.set_initial_value(r.y,r.t)
            r.integrate(next_t)
            collect_integrator_results(r.t, r.y.flatten(), out)
            
    # TODO: Figure out how to filter dense output to meshed output w/ dt controller
            

def SimulateControlledSystem(tspan, system, controller=None, x0=None):
    """
    system is a callable that takes current states and input and returns the
    first derivative the states and the outputs. (not actually implemented yet).
    system should have attributes: n_states, n_outputs, n_inputs, used for some
    setup.
    
    controller takes current time, (states,) and output, and returns the input.
    If None, will do an open loop simulation. 
    
    tspan is either the duration (or endpoints) of the simulation or the points 
    to evaluate the integral. If duration (number or single element list-like) 
    or as end points (two-element list-like), returns all points from numerical 
    integrator ("dense" output). If list of time-points, returns only those 
    points ("meshed") -- not implemented for DT systems.
    
    x0 is the initial state. If None uses x0 = 0 (vector of appropriate length).
    
    This function will handle the integration 
    """
    # This should handle all non-linear systems, even with LTI-based 
    # controller. In fact, I should write a linear controller class that can be
    # easily simulated. Truly linear systems can just be analyzed with Murray's
    # control. The linear controller class should be able to pull gains from 
    # that library (a NumPy matrix, I assume?)
    
    # TODO: Should I also have a controller base-class for controllers to
    # inherit? Or just stimulate fuzzy control systems?
    
    # TODO: fix for outputs
    # TODO: separate function_to_integrate constructor? User constructs feedback model?
    
    sim_result = SimulationResult(system, controller)
    
    if controller is None:
        def controller(*args):
            return np.zeros((system.n_inputs,1))
            
    if x0 is None:
        # x0 = np.zeros((n_sys_output,1))
        x0 = np.zeros((system.n_states,1))
    # TODO: I'll just trust that x0 is the right size/shape. Maybe add some 
    # helper code?
    if x0.shape != (system.n_states,1):
        x0 = np.concatenate((x0.reshape(x0.size,1), np.zeros((system.n_states - x0.size, 1))), 0)

    def function_to_integrate(t, x, result):
        # u = controller(t, y[f.n_states:f.n_outputs+f.n_states]) 
        # TODO: Once outputs are fixed, update controller to take outputs as
        # input args. "controller" should also include observer? or is that a
        # third issue?
        key = sim_result.get_key(t,x)
        if key not in result.u_history and not np.isnan(t):
            result.u_history[key] = controller(t, x, result) # history seems legit
        elif np.isnan(t):
            return np.nan*np.ones((system.n_states,1))
        try:
            u = result.u_history[key].copy()
        except:
            pass
        # TODO: figure out when I need to copy and when I don't
        # I know `x` coming from ode integrator will not be matrix, so I should 
        # cast it here to the right shape
        if system.n_inputs != 0:
            return system(x, u)
        else:
            return system(x)

    num_time_points = 1
    if system.dt is None:
        IntegrateContinuousTimeSystem(function_to_integrate, sim_result, tspan, x0)
    else:
        IntegrateDiscreteTimeSystem(function_to_integrate, sim_result, tspan, x0, system.dt)
        
    
    return sim_result
    