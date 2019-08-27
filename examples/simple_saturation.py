import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from simupy.systems.symbolic import DynamicalSystem, dynamicsymbols
from simupy import block_diagram
from simupy.array import Array, r_
from simupy.discontinuities import SwitchedOutput

BlockDiagram = block_diagram.BlockDiagram
int_opts = block_diagram.DEFAULT_INTEGRATOR_OPTIONS.copy()
find_opts = block_diagram.DEFAULT_EVENT_FIND_OPTIONS.copy()


int_opts['rtol'] = 1E-12
int_opts['atol'] = 1E-15
int_opts['nsteps'] = 1000

find_opts['xtol'] = 1E-12
find_opts['maxiter'] = int(1E3)

# This example shows how to implement a simple saturation block

# create an oscillator to generate the sinusoid
x = Array([dynamicsymbols('x')]) # placeholder output symbol
tvar = dynamicsymbols._t # use this symbol for time
sin = DynamicalSystem(Array([sp.cos(tvar)]), x) # define the oscillator,



llim = -0.75
ulim = 0.75
saturation_limit = r_[llim, ulim]
saturation_output = r_['0,2', llim, x[0], ulim]
sat = SwitchedOutput(x[0], saturation_limit, output_equations=saturation_output, input_=x)

sat_bd = BlockDiagram(sin, sat)
sat_bd.connect(sin, sat)

sat_res = sat_bd.simulate(2*np.pi, integrator_options=int_opts, event_find_options=find_opts)
plt.figure()
plt.plot(sat_res.t, sat_res.y[:, 0])
plt.plot(sat_res.t, sat_res.y[:, 1])

plt.xlabel('$t$, s')
plt.ylabel('$x(t)$')
plt.title('simple saturation demonstration')
plt.tight_layout()
plt.show()


eps = 0.25
deadband_limit = r_[-eps, eps]
deadband_output = r_['0,2', x[0]+eps, 0, x[0]-eps]
ded = SwitchedOutput(x[0], deadband_limit, output_equations=deadband_output, input_=x)
ded_bd = BlockDiagram(sin, ded)
ded_bd.connect(sin, ded)
ded_res = ded_bd.simulate(2*np.pi, integrator_options=int_opts, event_find_options=find_opts)

plt.figure()
plt.plot(ded_res.t, ded_res.y[:, 0])
plt.plot(ded_res.t, ded_res.y[:, 1])

plt.xlabel('$t$, s')
plt.ylabel('$x(t)$')
plt.title('simple deadband demonstration')
plt.tight_layout()
plt.show()
