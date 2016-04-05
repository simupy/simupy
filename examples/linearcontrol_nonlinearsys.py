import numpy as np, sympy as sp,  matplotlib.pyplot as plt, numpy.matlib
from NDMS.Systems import DynamicalSystem
from NDMS.Simulation import SimulateControlledSystem
from sympy.physics.mechanics import dynamicsymbols
import control
from NDMS.Matrices import construct_explicit_matrix, matrix_subs, matrix_callable_from_vector_trajectory, system_from_matrix_DE

x = sp.Matrix(dynamicsymbols('x1:4'))
u = dynamicsymbols('u')
x1, x2, x3 = x

sys = DynamicalSystem(sp.Matrix([-x1+x2-x3,-x1*x2-x2+u,-x1+u]),x,sp.Matrix([u]))
print(sys.jacobian().subs(matrix_subs([(x,np.zeros((3,1)))])))

A=np.matlib.mat("-1,1,-1;0,-1,0;-1,0,0") # linearization
B=np.matrix([0,1,1]).T

lqrres = control.lqr(A,B,np.matlib.eye(3),np.matrix([1]))
Kgain = np.matrix(lqrres[0])
legends = [r'$x_1$',r'$x_2$',r'$x_3$',r'$u$']

ctr = lambda t,x,*args: -Kgain*np.matrix(x).T

tF = 10

# case 1
x0 = np.matrix([1, 1, 2.25])
simres =  SimulateControlledSystem(tF, sys, ctr, x0)
plt.figure()
plt.plot(simres.t,simres.x,simres.t,simres.u)
plt.legend(legends)
plt.title('controlled system with unstable initial conditions')
plt.xlabel('time (s)')

# case 2
x0 = np.matrix([5, -3, 1])
simres =  SimulateControlledSystem(tF, sys, ctr, x0)
plt.figure(2)
plt.plot(simres.t,simres.x,simres.t,simres.u)
plt.legend(legends)
plt.title('controlled system with stable initial conditions')
plt.xlabel('time (s)')

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