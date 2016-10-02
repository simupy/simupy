import numpy as np, matplotlib.pyplot as plt, sympy as sp, numpy.matlib
from sympy.physics.mechanics import dynamicsymbols
from simupy.Systems import DynamicalSystem
from simupy.Simulation import SimulateControlledSystem
from simupy.utils import callable_from_trajectory
from simupy.Matrices import construct_explicit_matrix, matrix_subs, matrix_callable_from_vector_trajectory, system_from_matrix_DE


As = construct_explicit_matrix('a',2,2,dynamic=True)
Bs = construct_explicit_matrix('b',2,1,dynamic=True)
Qs = construct_explicit_matrix('q',2,2,symmetric=True,dynamic=True)
Rs = construct_explicit_matrix('r',1,1,symmetric=True,dynamic=True)
Ss = construct_explicit_matrix('s',2,2,symmetric=True,dynamic=True)
Gs = construct_explicit_matrix('g',2,1,dynamic=True)
rxs = construct_explicit_matrix('rx',2,1,dynamic=True)

Ssdot = (Qs+As.T*Ss+Ss*As-Ss*Bs*Rs.inv()*Bs.T*Ss)
Gsdot = As.T*Gs - Ss*Bs*Rs.inv()*Bs.T*Gs - Qs*rxs
#Gsdot = As.T*Gs  Ss*Bs*Rs.inv()*Bs.T*Gs - Qs*rxs
An = np.mat("0,1;-2,-3")
Bn = np.mat("0;1")
Qn = np.mat("1,0;0,0")
Rn = np.mat("0.02")

tF = 20
SGdot = Ssdot.row_join(Gsdot)
SG = Ss.row_join(Gs)

SG_sys = system_from_matrix_DE(SGdot, SG,  rxs, [(As,An),(Bs,Bn),(Qs,Qn),(Rs,Rn)])
ref_input_ctr = lambda t,*args: np.matrix([2*(tF-t),0]).T
sg_sim_res = SimulateControlledSystem(tF, SG_sys, ref_input_ctr, x0=np.matrix([0,0,0,0,0]).T )

mat_sg_result = matrix_callable_from_vector_trajectory(np.flipud(tF-sg_sim_res.t),np.flipud(sg_sim_res.x), SG_sys.states, SG)
vec_sg_result = matrix_callable_from_vector_trajectory(np.flipud(tF-sg_sim_res.t),np.flipud(sg_sim_res.x), SG_sys.states, SG_sys.states)

plt.plot() # Plot S components
plt.plot(sg_sim_res.t, mat_sg_result(sg_sim_res.t)[[0,0,1],[0,1,1],:].T)
plt.title('unforced component of solution to Riccatti differential equation')
plt.figure() # Plot G components
plt.plot(sg_sim_res.t, mat_sg_result(sg_sim_res.t)[:,-1,:].T)
plt.title('forced component of solution to Riccatti differential equation')

x = sp.Matrix(dynamicsymbols('x1:3'))
x1, x2 = x

sys = DynamicalSystem(
    As*x-Bs*Rs.inv()*Bs.T*(Ss*x+Gs), #state-equation
    x, #states
    SG_sys.states, #inputs
    constants_values=matrix_subs([(As,An),(Bs,Bn),(Qs,Qn),(Rs,Rn)]))
sysctrl = lambda t,*args: vec_sg_result(t)

syssimres = SimulateControlledSystem(tF, sys, sysctrl, x0=np.matrix([0,0]).T )
uu = np.einsum('ij,jk...->...',-Rn.I*Bn.T,np.einsum('ij...,jk...->ik...',mat_sg_result(syssimres.t)[:2,:2],syssimres.x.T[:,np.newaxis,:]) + mat_sg_result(syssimres.t)[:,2:])

plt.figure()
plt.plot(syssimres.t,syssimres.x,syssimres.t,2*syssimres.t)
plt.title('reference and states vs. time')
plt.xlabel('time (s)')
plt.legend([r'$x_1$',r'$x_2$',r'$r_1$'],loc=3)

plt.figure()
plt.plot(syssimres.t,uu)
plt.title('optimal control law vs. time')
plt.xlabel('time (s)')