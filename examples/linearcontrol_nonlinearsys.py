import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from simupy.systems import LTISystem
from simupy.systems.symbolic import DynamicalSystem, dynamicsymbols
from simupy.block_diagram import BlockDiagram
from sympy.tensor.array import Array
from numpy import matlib

legends = [r'$x_1(t)$', r'$x_2(t)$', r'$x_3(t)$', r'$u(t)$']
tF = 6

"""
This example shows the design of a linear quadratic regulator for a
nonlinear system linearized about the origin. It is stable for some initial
conditions, but not all initial conditions. The region of stability is not
dependent only on the distance from the origin.
"""

# construct system
x = Array(dynamicsymbols('x1:4'))
u = dynamicsymbols('u')
x1, x2, x3 = x
sys = DynamicalSystem(Array([-x1+x2-x3, -x1*x2-x2+u, -x1+u]), x, Array([u]))

# linearization to design LQR
t0 = 0
x0 = np.zeros((3, 1))
u0 = 0
A = sys.state_jacobian_equation_function(t0, x0, u0)
B = sys.input_jacobian_equation_function(t0, x0, u0)

# LQR gain
Q = np.matlib.eye(3)
R = np.matrix([1])
S = linalg.solve_continuous_are(A, B, Q, R,)
K = linalg.solve(R, B.T @ S).reshape(1, -1)
ctr_sys = LTISystem(-K)

# Construct block diagram
BD = BlockDiagram(sys, ctr_sys)
BD.connect(sys, ctr_sys)
BD.connect(ctr_sys, sys)

# case 1 - un-recoverable
sys.initial_condition = np.r_[1, 1, 2.25]
result1 = BD.simulate(tF)
plt.figure()
plt.plot(result1.t, result1.y)
plt.legend(legends)
plt.title('controlled system with unstable initial conditions')
plt.xlabel('$t$, s')
plt.tight_layout()
plt.show()

# case 2 - recoverable
sys.initial_condition = np.r_[5, -3, 1]
result2 = BD.simulate(tF)
plt.figure()
plt.plot(result2.t, result2.y)
plt.legend(legends)
plt.title('controlled system with stable initial conditions')
plt.xlabel('$t$, s')
plt.tight_layout()
plt.show()
