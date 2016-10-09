# SimuPy

A framework for modeling and simulating dynamical systems.

## Overview

The goal of SimuPy is to provide a framework for simulating inter-connected dynamical system models. Models can be constructed using symbolic expressions, as in

```python
from simupy.Systems import DynamicalSystem
x = x1, x2, x3 = sp.Matrix(dynamicsymbols('x1:4'))
u = dynamicsymbols('u')
sys = DynamicalSystem(sp.Matrix([-x1+x2-x3, -x1*x2-x2+u, -x1+u]), x, u)
```

which will automatically create callable functions for the state equations, output equations, and jacobians. By default, the code generator uses a wrapper for ``sympy.lambdify``. You can change it by passing the system initialization arguments ``code_generator`` (the function) and additional key-word arguments to the generator in a dictionary ``code_generator_args``. You can change the defaults for future systems by changing the module values

``python
import simupy.Systems
simupy.Systems.DEFAULT_CODE_GENERATOR = your_code_generator_function
simupy.Systems.DEFAULT_CODE_GENERATOR_ARGS = {'extra_arg': value}
``

A number of helper classes/functions exist to simplify the construction of models. For example, a linear feedback controller can be defined as

```python
from simupy.Systems import LTISystem
ctrl = LTISystem(matrix([[-1.73992128, -0.99212953,  2.98819041]]))
```

(the gains in the example come from the infinite horizon LQR based on the system linearized about the origin.) A block diagram of the feedback control can be constructed

```python
from simupy.BlockDiagram import BlockDiagram
BD = BlockDiagram(sys, ctrl)
BD.connect(sys, ctrl) # connect the state current states to the feedback controller
BD.connect(ctrl, sys) # connect the controlled input to the system
```

Initial conditions for systems with non-zero state can be defined and the interconnected systems can be simulated

```python
sys.initial_condition = np.matrix([5, -3, 1])
res = BD.simulate(10)
```

which uses ``scipy.integrate.ode`` to solve the initial-valued problem. The results are an instance of the ``SimulationResult`` class, with properties ``t``, ``x``, and ``y``, holding ``N`` long array of time indices, an ``N x n`` array of of state values at each time, and ``N x p`` array of output values at each time, respectively. The simulation defaults to the ``dopri5`` solver with dense output, but other solvers and solver options can be passed. 

A number of utilities for constructing and manipulating systems and the simulation results are also included:

- ``process_vector_args`` and ``lambdify_with_vector_args`` from ``simupy.utils`` are helpers for code generation using ``sympy.lambdify``
- ``simupy.utils.callable_from_trajectory`` is a simple wrapper for making polynomial spline interpolators using ``scipy.interpolate.splprep``
- ``simupy.Matrices`` includes tools for constructing (vector) systems using matrix expressions and re-wrapping the results into matrix form
- ``simupy.Systems.SystemFromCallable`` is a helper for converting a function to a state-less system (typically controller) to simulate
- ``MemorylessSystem`` and ``LTISystem`` are subclasses to more quickly create these types of systems
- ``DescriptorSystem`` is used to construct systems with dynamics of the form ``M(t, x) * x'(t) = f(t,x,u)``. In the future, this form can be used in DAE solvers, etc



## Design

SimuPy assumes systems have no direct feedthrough between inputs and outputs. To simulate a system model that includes a feedthrough, the system can be augmented (differentiating the input or integrating the output). However, there is no requirement for the system to have a state, so 

```
x'(t) = f(t,x,u)
y(t) = h(t,x)
```

and 

```
y(t) = h(t,u)
```

are both valid formulations. A system in a ``BlockDiagram`` needs to provide ``n_states``, ``n_inputs``, ``n_outputs``, ``output_equation_function``. If ``n_states`` > 0 then ``state_equation_function`` must also be provided. In the future, providing jacobian functions will be used to construct ``BlockDiagram`` jacobians to use with solvers that support them.

Setting system property ``dt``>0 will determine the sample rate that the outputs and states are computed; ``dt``=0 is treated as a continuous-time system. In hybrid-time ``BlockDiagram``s, the system is automatically integrated piecewise to improve accuracy. Assumes systems are defined as

```
x[k+] = f([k],x(k),u(k)])
y[k+] = h([k],x[k+])
```

and

```
y[k+] = h([k], u(k))
```

where ``[k]`` and ``(k)`` the value of the variable at time ``k*dt`` for discrete-time and continuous time systems respectively, while ``[k+]`` indicates the value of variable over the interval ``(k*dt, (k+1)*dt]``. This should have the expected result that a block diagram with only discrete-time systems behaves like 

```
x[k+1] = f([k], x[k], u[k])
y[k] = h([k], x[k])
```

and makes sense in general for hybrid-time simulation.

By choice, control design is outside the scope of SimuPy. So controller design tools (for example, feedback linearization, sliding mode, "adapative", etc) should be in its own library, but analysis tools that might help in controller design could be appropriate here (Lie Algebra, features described in future goals, etc)

## Future goals
- Add new code generator/wrappers (theano, ufuncify, ?). First 
- Make sure there is a consistent convention for ordering. I think this is important for the trajectory callable (time, states) including matrices, as well as jacobians and vectorized state/output/jacobian functions inputs and outputs. 
- Hooks to allow integrators other than ``scipy.integrate.ode`` including DAE solvers and/or event handling and/or homeotopic methods for discontinuities
- Construct BlockDiagram level jacobians (of the continuous systems)
- Linear and Non-linear system analysis tools, such as
  - helpers for Lyapunov analysis
  - phase plane plotters
  - describing functions, 
  - stability, controllability, observability, frequency response
- Tools to manipulate and present the results more easily:
  - a ``Pandas`` subclass:
    - selectors based on time and state conditions (end-points are interpolated)
    - algebraic manipulation to create new columns
    - convert sympy expressions when naming or getting columns
    - tool to plot reults together (collect on column names)
  - plotting tools to compare results from similar simulation runs
- special constants dict that tracks Systems, updates on key change.