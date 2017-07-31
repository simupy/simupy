import numpy as np, matplotlib.pyplot as plt, sympy as sp, numpy.matlib
from sympy.physics.mechanics import dynamicsymbols
from simupy.systems import DynamicalSystem, MemorylessSystem, SystemFromCallable, LTISystem
from simupy.utils import callable_from_trajectory
from simupy.block_diagram import BlockDiagram

plt.ion()

x = sp.Matrix([dynamicsymbols('x')])
tvar = dynamicsymbols._t

sin = DynamicalSystem(sp.Matrix([sp.cos(tvar)]), x)

sin_bd = BlockDiagram(sin)
sin_res = sin_bd.simulate(2*np.pi)
plt.plot(sin_res.t,sin_res.x)

from simupy.discontinuities import SwitchedOutput


sat = SwitchedOutput(x[0,0], [-0.75, 0.75], sp.Matrix([-0.75, x[0,0], 0.75]), x)
sat_bd = BlockDiagram(sin, sat)
sat_bd.connect(sin, sat)
sat_res = sat_bd.simulate(2*np.pi)#, integrator_options={'rtol': 1E-9})
plt.plot(sat_res.t, sat_res.y[:,-1])