import numpy as np, matplotlib.pyplot as plt, sympy as sp, numpy.matlib
from sympy.physics.mechanics import dynamicsymbols
from simupy.systems import DynamicalSystem, SystemFromCallable, LTISystem
from simupy.utils import callable_from_trajectory, array_callable_from_vector_trajectory
from simupy.block_diagram import BlockDiagram
from simupy.matrices import construct_explicit_matrix, matrix_subs, system_from_matrix_DE

plt.ion()

As = construct_explicit_matrix('a',2,2,dynamic=False)
Bs = construct_explicit_matrix('b',2,1,dynamic=False)
Qs = construct_explicit_matrix('q',2,2,symmetric=True,dynamic=False)
Rs = construct_explicit_matrix('r',1,1,symmetric=True,dynamic=False)
Ss = construct_explicit_matrix('s',2,2,symmetric=True,dynamic=True)
Gs = construct_explicit_matrix('g',2,1,dynamic=True)
rxs = construct_explicit_matrix('rx',2,1,dynamic=True)

Ssdot = (Qs+As.T*Ss+Ss*As-Ss*Bs*Rs.inv()*Bs.T*Ss)
Gsdot = As.T*Gs - Ss*Bs*Rs.inv()*Bs.T*Gs - Qs*rxs
An = np.mat("0,1;-2,-3")
Bn = np.mat("0;1")
Qn = np.mat("1,0;0,0")
Rn = np.mat("0.02")

tF = 20
SGdot = Ssdot.row_join(Gsdot)
SG = Ss.row_join(Gs)
SG_subs = dict(matrix_subs((As,An),(Bs,Bn),(Qs,Qn),(Rs,Rn)))

SG_sys = system_from_matrix_DE(SGdot, SG,  rxs, SG_subs)
ref_input_ctr = lambda t,*args: np.r_[2*(tF-t),0]
ref_input_ctr_sys = SystemFromCallable(ref_input_ctr, 0, 2)

RiccatiBD = BlockDiagram(SG_sys, ref_input_ctr_sys)
RiccatiBD.connect(ref_input_ctr_sys, SG_sys)
sg_sim_res = RiccatiBD.simulate(tF)

sim_data_unique_t, sim_data_unique_t_idx = np.unique(sg_sim_res.t, return_index=True)

mat_sg_result = array_callable_from_vector_trajectory(np.flipud(tF-sg_sim_res.t[sim_data_unique_t_idx]),np.flipud(sg_sim_res.x[sim_data_unique_t_idx]), SG_sys.state, SG)
vec_sg_result = array_callable_from_vector_trajectory(np.flipud(tF-sg_sim_res.t[sim_data_unique_t_idx]),np.flipud(sg_sim_res.x[sim_data_unique_t_idx]), SG_sys.state, SG_sys.state)

plt.plot() # Plot S components
plt.plot(sg_sim_res.t, mat_sg_result(sg_sim_res.t)[:,[0,0,1],[0,1,1]])
plt.title('unforced component of solution to Riccatti differential equation')
plt.figure() # Plot G components
plt.plot(sg_sim_res.t, mat_sg_result(sg_sim_res.t)[:,:,-1])
plt.title('forced component of solution to Riccatti differential equation')

##
tracking_controller = SystemFromCallable(lambda t, xx: -Rn**-1@Bn.T@(mat_sg_result(t)[:,:-1]@xx + mat_sg_result(t)[:,-1]), 2, 1)
sys = LTISystem(An, Bn)

sys.initial_condition = np.zeros((2,1))
# int_opts = block_diagram.DEFAULT_INTEGRATOR_OPTIONS.copy()
# int_opts['rtol'] = 1E-15
# int_opts['atol'] = 1E-15
# int_opts['max_step'] = 0.25

control_BD = BlockDiagram(sys, tracking_controller)
control_BD.connect(sys, tracking_controller)
control_BD.connect(tracking_controller, sys)
control_res = control_BD.simulate(tF) #, integrator_options=int_opts)

plt.figure()
plt.plot(control_res.t,control_res.x,control_res.t,2*control_res.t)
plt.title('reference and state vs. time')
plt.xlabel('time (s)')
plt.legend([r'$x_1$',r'$x_2$',r'$r_1$'],loc=3)

plt.figure()
plt.plot(control_res.t,control_res.y[:,-1])
plt.title('optimal control law vs. time')
plt.xlabel('time (s)')
