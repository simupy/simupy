import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from simupy.systems.symbolic import DynamicalSystem, dynamicsymbols
from simupy.block_diagram import BlockDiagram, DEFAULT_INTEGRATOR_OPTIONS
from simupy.array import Array, r_

# This example simulates the Van der Pol oscillator.

DEFAULT_INTEGRATOR_OPTIONS['nsteps'] = 1000

x = x1, x2 = Array(dynamicsymbols('x1:3'))

mu = sp.symbols('mu')

state_equation = r_[x2, -x1+mu*(1-x1**2)*x2]
output_equation = r_[x1**2 + x2**2, sp.atan2(x2, x1)]

sys = DynamicalSystem(
    state_equation,
    x,
    output_equation=output_equation,
    constants_values={mu: 5}
)

sys.initial_condition = np.array([1, 1]).T

BD = BlockDiagram(sys)
res = BD.simulate(30)

plt.figure()
plt.plot(res.t, res.x)
plt.legend([sp.latex(s, mode='inline') for s in sys.state])
plt.ylabel('$x_i(t)$')
plt.xlabel('$t$, s')
plt.title('system state vs time')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(*res.x.T)
plt.xlabel('$x_1(t)$')
plt.ylabel('$x_2(t)$')
plt.title('phase portrait of system')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(res.t, res.y)
plt.legend([r'$\left| \mathbf{x}(t) \right|$', r'$\angle \mathbf{x} (t)$'])
plt.xlabel('$t$, s')
plt.title('system outputs vs time')
plt.tight_layout()
plt.show()
