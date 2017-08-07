API Documentation
=================

A system in a ``BlockDiagram`` needs to provide the following attributes:

    - ``dim_state`` : the dimension of the state
    - ``dim_input`` : the dimension of the input
    - ``dim_output`` : the dimension of the output
    - ``dt`` : the sampling rate of the system; 0 for continuous time systems.
    - ``output_equation_function`` : A callable returning the system output.

If ``dim_state`` == 0, then ``output_equation_function`` recieves the current time and input as arguments during integration. If ``dim_state`` > 0 then ``state_equation_function``, taking the current time, state, and input and returning the state derivative, must also be provided. In this case, ``output_equation_function`` recieves the current time and state as arguments during integration.

If ``event_equation_function`` and ``update_equation_function`` are provided, discontinuities at zero-crossing of ``event_equation_function`` are handled. The argument rules for ``event_equation_function`` and ``update_equation_function`` during integration are the same as ``output_equation_function``. Generally, ``update_equation_function`` is used to change what ``state_equation_function``, ``output_equation_function``, and ``event_equation_function`` compute based on the occurance of the discontinuity. If ``dim_state`` > 0, ``update_equation_function`` must return the state immediately after the discontinuity.

Setting system property ``dt`` > 0 will determine the sample rate that the outputs and state are computed; ``dt`` = 0 is treated as a continuous-time system. In hybrid-time ``BlockDiagram``s, the system is automatically integrated piecewise to improve accuracy.

In the future, providing jacobian functions will be used to construct ``BlockDiagram`` jacobians to use with solvers that support them. 

The generated API documents below come from the docstrings:

.. toctree::
   :maxdepth: 4

   array
   block_diagram
   descriptor
   discontinuities
   matrices
   systems
   utils
