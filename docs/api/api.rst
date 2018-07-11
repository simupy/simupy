API Documentation
=================

A system in a ``BlockDiagram`` needs to provide the following attributes:

    - ``dim_state`` : the dimension of the state
    - ``dim_input`` : the dimension of the input
    - ``dim_output`` : the dimension of the output
    - ``output_equation_function`` : A callable returning the system output.

If ``dim_state``\=0, then ``output_equation_function`` recieves the current
time and input as arguments during integration. If ``dim_state``\>0 then
``state_equation_function``, taking the current time, state, and input and
returning the state derivative, must also be provided. In this case,
``output_equation_function`` recieves the current time and state as arguments
during integration.

If ``event_equation_function`` and ``update_equation_function`` are provided,
discontinuities at zero-crossing of ``event_equation_function`` are handled.
The argument rules for ``event_equation_function`` and 
``update_equation_function`` during integration are the same as for 
``output_equation_function`` and ``state_equation_function``, respectively.
Generally, ``update_equation_function`` is used to change what
``state_equation_function``, ``output_equation_function``, and
``event_equation_function`` compute based on the occurance of the
discontinuity. If ``dim_state``\>0, ``update_equation_function`` must return
the state immediately after the discontinuity.

The base system class takes a convenience input argument, ``dt``. Passing
``dt``\>0 will determine the sample rate that the
outputs and state are computed; ``dt``\=0 is treated as a continuous-time
system. In hybrid-time ``BlockDiagram``\s, the system is automatically
integrated piecewise to improve accuracy.

Future versions of SimuPy may support passing jacobian functions to ode solvers 
if all systems in the ``BlockDiagram`` provide the appropriate necessary
jacobian functions.

A quick overview of the of the modules:

``block_diagram`` (:doc:`docstrings<block_diagram>`)
   implements the ``BlockDiagram`` class to simulate interconnected systems.
``systems`` (:doc:`docstrings<systems>`)
   provides a few base classes for purely numerical based systems.
``utils`` (:doc:`docstrings<utils>`)
   provides utility functions, such as manipulating (numeric) systems and
   simulation results.
``systems.symbolic`` (:doc:`docstrings<symbolic_systems>`) and ``discontinuities`` (:doc:`docstrings<discontinuities>`)
   provides niceties for using symbolic expressions to define systems.
``array`` (:doc:`docstrings<array>`) and ``matrices`` (:doc:`docstrings<matrices>`)
   provide helper functions and classes for manipulating symbolic arrays,
   matrices, and their systems.
``utils.symbolic`` (:doc:`docstrings<symbolic_utils>`)
   provides utility symbolic functions, such as manipulating symbolic systems.

.. toctree::
   :hidden:
   :maxdepth: 4

   block_diagram
   systems
   utils
   symbolic_systems
   discontinuities
   array
   matrices
   symbolic_utils
