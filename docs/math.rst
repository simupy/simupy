Mathematical Formulation
========================

SimuPy assumes systems have no direct feedthrough between inputs and outputs;
this discpline avoids algebraic loops. You can simulate a system model that
includes a feedthrough by augmenting the system. Augment the system using the
input by including input components in the state and using derivatives of those
signals in the control input. You can augment the system using the output by
including the original output components in the state and using integrals of
those signals in the system output. However, there is no requirement for the
system to have a state, so

.. math::
    x'(t) &= f(t,x(t),u(t)) \\
    y(t) &= h(t,x(t))


and

.. math::
    y(t) = h(t,u(t))


are both valid formulations. Here, :math:`t` is the time variable, :math:`x`
is the system state, :math:`u` is the system input, :math:`y` is the sytem
output. We call :math:`f` the state equation and :math:`h` the output equation.
SimuPy can also handle discrete time systems of the form


.. math::
    x[k+1] &= f([k],x[k],u(k)]) \\
    y[k+1] &= h([k],x[k+1])

and

.. math::
    y[k+1] = h([k], u(k))


where :math:`[k]` indicates signal values over the half-open interval 
:math:`(k\, dt, (k+1) dt]` which are updated at time :math:`t=k\, dt` for
discrete-time systems and :math:`(k)` indicates a zero-order hold sample of the
signal at time :math:`k \, dt` for continuous-time systems. This formulation
gives the expected results for models with only discrete-time sub-systems of
the same update rate :math:`dt` which can be combined into a single system 
of the form

.. math::
    x[k+1] &= f([k], x[k], u[k]) \\
    y[k] &= h([k], x[k])

and makes sense in general for hybrid-time simulation. 

This formulation is also consistent with common linear, time-invariant (LTI)
system algebras and transformations. For example, the dynamics of the LTI
system

.. math::
    x'(t) &= A \, x(t) + B \, u(t)
    y(t) &= I \, x(t)

with state-feedback

.. math::
    u(t) = -K\, x(t)

are the same as the autonomous system

.. math::
    x'(t) &= (A - B\,K) \, x(t)
    y(t) &= I \, x(t).

as demonstrated in the tests, ``test_feedback_equivalent``.

The validity of the
timing and interconnection defintions is shown via tests in the
``test_block_diagram.py`` test file, including ``test_feedback_equivalent``,
``test_dt_ct_equivalent``, and ``test_mixed_dts``. It is also illustrated in
the ``discrete_lti.py`` example.
