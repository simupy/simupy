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
is the system state, :math:`u` is the system input, and :math:`y` is the sytem
output. We call :math:`f` the state equation and :math:`h` the output equation.
SimuPy can also handle discrete-time systems with sample period :math:`\Delta t`
of the form

.. math::
    x[k+1] &= f([k],x[k],u(k)]) \\
    y[k+1] &= h([k+1],x[k+1])

and

.. math::
    y[k] = h([k], u(k))


where :math:`[k]` indicates signal values over the half-open interval 
:math:`(k\, \Delta t, (k+1) \Delta t]` which are updated at time 
:math:`t=k\, \Delta t` for discrete-time systems and :math:`(k)` indicates a 
zero-order hold sample of the signal at time :math:`k \, \Delta t` for 
continuous-time systems. This formulation gives the expected results for models
with only discrete-time sub-systems of the same update rate :math:`\Delta t` 
which can be combined into a single system of the form

.. math::
    x[k+1] &= f([k], x[k], u[k]) \\
    y[k] &= h([k], x[k])

and makes sense in general for hybrid-time simulation. 

This formulation is also consistent with common linear, time-invariant (LTI)
system algebras and transformations. For example, the dynamics of the LTI
system

.. math::
    x'(t) &= A \, x(t) + B \, u(t), \\
    y(t) &= I \, x(t),

with state-feedback

.. math::
    u(t) = -K\, x(t),

are the same as the autonomous system

.. math::
    x'(t) &= (A - B\,K) \, x(t), \\
    y(t) &= I \, x(t).

Similarly, timing transformations are consistent. The discrete-time equivalent
of the continuous-time LTI system above,

.. math::
    x[k+1] &= \Phi\, x[k] + \Gamma\, u[k], \\
    y[k] &= I \, x[k],

will travel through the same state trajectory at times :math:`k\, \Delta t` if
both are subject to the same piecewise constant inputs and the state and input 
matrices are related by the zero-order hold transformation

.. math::
    \Phi &= e^{A\, \Delta t}, \\
    \Gamma &= \int_{0}^{\Delta t} e^{A\, \tau} \, d \tau B.

The accuracy of these algebras and transformations are demonstrated in the
``discrete_lti.py`` example and are incorporated into the
``test_block_diagram.py`` tests.
