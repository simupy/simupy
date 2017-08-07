# SimuPy

A framework for modeling and simulating dynamical systems. This provides an open source, Python based alternative to Simulink.

## Installation



## Overview

The goal of SimuPy is to provide a framework for simulating inter-connected dynamical system models. Models can be constructed using symbolic expressions, as in

```python
from simupy.Systems import DynamicalSystem
x = x1, x2, x3 = sp.Matrix(dynamicsymbols('x1:4'))
u = dynamicsymbols('u')
sys = DynamicalSystem(sp.Matrix([-x1+x2-x3, -x1*x2-x2+u, -x1+u]), x, u)
```

which will automatically create callable functions for the state equations, output equations, and jacobians. By default, the code generator uses a wrapper for ``sympy.lambdify``. You can change it by passing the system initialization arguments ``code_generator`` (the function) and additional key-word arguments to the generator in a dictionary ``code_generator_args``. You can change the defaults for future systems by changing the module values

```python
import simupy.systems
simupy.systems.DEFAULT_CODE_GENERATOR = your_code_generator_function
simupy.systems.DEFAULT_CODE_GENERATOR_ARGS = {'extra_arg': value}
```

A number of helper classes/functions exist to simplify the construction of models. For example, a linear feedback controller can be defined as

```python
from simupy.systems import LTISystem
ctrl = LTISystem(matrix([[-1.73992128, -0.99212953,  2.98819041]]))
```

(the gains in the example come from the infinite horizon LQR based on the system linearized about the origin.) A block diagram of the feedback control can be constructed

```python
from simupy.block_diagram import BlockDiagram
BD = BlockDiagram(sys, ctrl)
BD.connect(sys, ctrl) # connect the current state to the feedback controller
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
- ``DiscontinuousSystem`` is used to construct systems with discontinuities, defined by zero-crossings of the ``event_equation_function`` output.

