import numpy as np, sympy as sp,  matplotlib.pyplot as plt, numpy.matlib
from NDMS.Systems import DynamicalSystem, SystemFromCallable
from NDMS.BlockDiagram import BlockDiagram
from NDMS.utils import grad
from sympy.physics.mechanics import dynamicsymbols
import control
from NDMS.Matrices import construct_explicit_matrix, matrix_subs, matrix_callable_from_vector_trajectory, system_from_matrix_DE

x = sp.Matrix(dynamicsymbols('x1:4'))
u = dynamicsymbols('u')
x1, x2, x3 = x

sys = DynamicalSystem(sp.Matrix([-x1+x2-x3,-x1*x2-x2+u,-x1+u]),x,sp.Matrix([u]))
print(sys.jacobian().subs(matrix_subs([(x,np.zeros((3,1)))])))

A=np.matlib.mat("-1,1,-1;0,-1,0;-1,0,0") # linearization
# B=np.matrix([0,1,1]).T

A = np.matrix(grad(sys.state_equations,x).subs(matrix_subs([(x,np.zeros((3,1)))])).tolist(),dtype=np.float_)
B = np.matrix(grad(sys.state_equations,[u]).subs(matrix_subs([(x,np.zeros((3,1)))])).tolist(),dtype=np.float_)

lqrres = control.lqr(A,B,np.matlib.eye(3),np.matrix([1]))
Kgain = np.matrix(lqrres[0])
legends = [r'$x_1$',r'$x_2$',r'$x_3$',r'$u$']

ctr_call = lambda t,x,*args: -Kgain*np.matrix(x).T
ctr_sys = SystemFromCallable(ctr_call, 3, 1)

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
plt.figure(2)
plt.plot(result2.t,result2.y)
plt.legend(legends)
plt.title('controlled system with stable initial conditions')
plt.xlabel('time (s)')

"""
# case 3
tF = 5
ctr = lambda *args: np.matrix([0.0])
x0 = np.matrix([5, -3, 1])
simres =  SimulateControlledSystem(tF, sys, ctr, x0)
plt.figure(3)
plt.plot(simres.t,simres.x,simres.t,simres.u)
plt.legend(legends)
plt.title('uncontrolled system')
plt.xlabel('time (s)')
"""