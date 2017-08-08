import numpy as np, sympy as sp,  matplotlib.pyplot as plt, numpy.matlib
from simupy.systems.symbolic import DynamicalSystem
from simupy.block_diagram import BlockDiagram
from sympy.physics.mechanics import dynamicsymbols
from sympy.tensor.array import Array

plt.ion()

x = Array(dynamicsymbols('x1:3'))
x1, x2 = x

mu = sp.symbols('mu')

sys = DynamicalSystem( Array([x2, -x1+mu*(1-x1**2)*x2]), x, constants_values={mu: 5})

sys.initial_condition = np.array([1,1]).T

BD = BlockDiagram(sys)
res = BD.simulate(30)

plt.plot(res.t,res.y)
plt.legend([sp.latex(s, mode='inline') for s in sys.state])

plt.figure()
plt.plot(*res.y.T)

from simupy.array import r_, c_
r_[x1,x2]