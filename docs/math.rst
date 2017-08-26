Mathematical Formulation
========================

SimuPy assumes systems have no direct feedthrough between inputs and outputs;
this discpline avoids algebraic loops. You can simulate a system model that
includes a feedthrough by augmenting the system. Augment the system using the
input by including input componets in the state and using derivatives of those
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
    x[k+] &= f([k],x[k],u(k)]) \\
    y[k+] &= h([k],x[k+])

and

.. math::
    y[k+] = h([k], u(k))


where :math:`[k]` and :math:`(k)` indexes the value of the variable at time
:math:`k \, dt` for discrete-time and continuous-time systems respectively,
while :math:`[k+]` indicates the value of variable over the interval :math:`(k
\, dt, (k+1) dt]`. This should have the expected result that a block diagram
with only discrete-time systems of the same sampling rate :math:`dt` which can
be combined into a single system of the form

.. math::
    x[k+1] &= f([k], x[k], u[k]) \\
    y[k] &= h([k], x[k])

and makes sense in general for hybrid-time simulation. The validity of the
timing and interconnection defintions is shown via tests in the
``test_block_diagram.py`` test file, including ``test_feedback_equivalent``,
``test_dt_ct_equivalent``, and ``test_mixed_dts``. It is also illustrated in
the ``discrete_lti.py`` example.
