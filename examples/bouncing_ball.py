import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from simupy.discontinuities import SwitchedSystem
from simupy import block_diagram
from simupy.array import Array, r_

BlockDiagram = block_diagram.BlockDiagram
int_opts = block_diagram.DEFAULT_INTEGRATOR_OPTIONS.copy()
find_opts = block_diagram.DEFAULT_EVENT_FIND_OPTIONS.copy()

"""
This example shows how to use a SwitchedSystem to model a bouncing ball. The
event detection accurately finds the point of impact, and the simulation
is generally accurate when the ball has sufficient energy. However, due to
numerical error, the simulation does show the ball chattering after all
the energy should have been dissipated.
"""

int_opts['rtol'] = 1E-12
int_opts['atol'] = 1E-15
int_opts['nsteps'] = 1000
int_opts['max_step'] = 2**-3

find_opts['xtol'] = 1E-12
find_opts['maxiter'] = int(1E3)


x = x1, x2 = Array(dynamicsymbols('x_1:3'))
mu, g = sp.symbols('mu g')
constants = {mu: 0.8, g: 9.81}
ic = np.r_[10, 15]
sys = SwitchedSystem(
    x1, Array([0]),
    state_equations=r_[x2, -g],
    state_update_equation=r_[sp.Abs(x1), -mu*x2],
    state=x,
    constants_values=constants,
    initial_condition=ic
)
bd = BlockDiagram(sys)
res = bd.simulate(
    20.36, integrator_options=int_opts, event_find_options=find_opts
)

expr_subs = constants.copy()
expr_subs[x1] = ic[0]
expr_subs[x2] = ic[1]
v1 = sp.sqrt(x2**2 + 2*g*x1).evalf(subs=expr_subs)
tstar = ((x2 + v1*(1 + mu)/(1-mu))/g).evalf(subs=expr_subs)

tvar = dynamicsymbols._t
impact_eq = (x2*tvar - g*tvar**2/2 + x1).subs(expr_subs)
t_impact = sp.solve(impact_eq, tvar)[-1]

# tstar is where the ball should come to a rest, however due to numerical
# error, it continues to chatter.
t_sel = (res.t < tstar*1.01)

plt.figure()
plt.subplot(2, 1, 1)
plt.title('bouncing ball')
plt.subplot(2, 1, 1)
plt.plot(res.t[t_sel], res.x[t_sel, 0])
plt.plot(2*[t_impact], [0, np.max(res.x[t_sel, 0])])
plt.ylabel('ball position, m')
plt.subplot(2, 1, 2)
plt.plot(res.t[t_sel], res.x[t_sel, 1])
plt.ylabel('ball velocity, m/s')
plt.xlabel('time, s')
plt.tight_layout()
plt.show()


plt.figure()
plt.subplot(2, 1, 1)
plt.title('bouncing ball chatter')
t_sel = (res.t > 20) & (res.t < tstar*1.03)
plt.subplot(2, 1, 1)
plt.plot(res.t[t_sel], res.x[t_sel, 0])
plt.ylabel('ball position, m')
plt.subplot(2, 1, 2)
plt.plot(res.t[t_sel], res.x[t_sel, 1])
plt.ylabel('ball velocity, m/s')
plt.xlabel('time, s')
plt.tight_layout()
plt.show()
