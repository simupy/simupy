import numpy as np, sympy as sp,  matplotlib.pyplot as plt, numpy.matlib
from simupy.Systems import DynamicalSystem, SystemFromCallable
from simupy.BlockDiagram import BlockDiagram
from sympy.physics.mechanics import dynamicsymbols

x = sp.Matrix(dynamicsymbols('x1:3'))
x1, x2 = x

mu = sp.symbols('mu')

sys = DynamicalSystem( sp.Matrix([x2, -x1+mu*(1-x1**2)*x2]), x, constants_values={mu: 5})

sys.initial_condition = np.matrix([1,1])

BD = BlockDiagram(sys)
res = BD.simulate(10)

plt.plot(res.t,res.y)
plt.legend([sp.latex(s, mode='inline') for s in sys.states])