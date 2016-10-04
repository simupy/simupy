import numpy as np, sympy as sp,  matplotlib.pyplot as plt, numpy.matlib
from simupy.Systems import DynamicalSystem, SystemFromCallable, LTISystem
from simupy.BlockDiagram import BlockDiagram
from simupy.utils import grad
from sympy.physics.mechanics import dynamicsymbols
import control

plt.ion()
x = sp.Matrix(dynamicsymbols('x1:4'))
u = dynamicsymbols('u')
x1, x2, x3 = x

t0 = 0
x0 = np.zeros((3,1))
u0 = 0

sys = DynamicalSystem(sp.Matrix([-x1+x2-x3,-x1*x2-x2+u,-x1+u]),x,sp.Matrix([u]))

A = sys.state_jacobian_equation_function(t0,x0,u0) # linearization
B = sys.input_jacobian_equation_function(t0,x0,u0)

Q = np.matlib.eye(3)
R = np.matrix([1])

# TODO: add eigendecomposition of Hamiltonian for LQR gain?
lqrres = control.lqr(A,B,Q,R)
Kgain = np.matrix(lqrres[0])
legends = [r'$x_1$',r'$x_2$',r'$x_3$',r'$u$']

ctr_call = lambda t,x,*args: -Kgain*np.matrix(x).T
ctr_sys = LTISystem(-Kgain) # SystemFromCallable(ctr_call, 3, 1)

BD = BlockDiagram(sys,ctr_sys)

BD.connect(sys,ctr_sys)
BD.connect(ctr_sys,sys)

tF = 10

# case 1
sys.initial_condition = np.matrix([1, 1, 2.25]).T
result1 = BD.simulate(tF)

plt.figure()
plt.plot(result1.t,result1.y)
plt.legend(legends)
plt.title('controlled system with unstable initial conditions')
plt.xlabel('time (s)')

# case 2
sys.initial_condition = np.matrix([5, -3, 1]).T
result2 = BD.simulate(tF)
plt.figure()
plt.plot(result2.t,result2.y)
plt.legend(legends)
plt.title('controlled system with stable initial conditions')
plt.xlabel('time (s)')