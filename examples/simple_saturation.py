import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from simupy.systems.symbolic import DynamicalSystem, dynamicsymbols
from simupy.block_diagram import BlockDiagram
from simupy.array import Array, r_
from simupy.discontinuities import SwitchedOutput

plt.ion()

llim = -0.75
ulim = 0.75

x = Array([dynamicsymbols('x')])
tvar = dynamicsymbols._t
sin = DynamicalSystem(Array([sp.cos(tvar)]), x)

sin_bd = BlockDiagram(sin)
sin_res = sin_bd.simulate(2*np.pi)

plt.figure()
plt.plot(sin_res.t, sin_res.x)

limit = r_[llim, ulim]
saturation_output = r_['0,2', llim, x[0], ulim]

sat = SwitchedOutput(x[0], limit, output_equations=saturation_output, input_=x)
sat_bd = BlockDiagram(sin, sat)
sat_bd.connect(sin, sat)
sat_res = sat_bd.simulate(2*np.pi)
plt.plot(sat_res.t, sat_res.y[:, -1])

plt.xlabel('$t$, s')
plt.ylabel('$x(t)$')
plt.title('simple saturation demonstration')
plt.tight_layout()
