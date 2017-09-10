import numpy as np
import matplotlib.pyplot as plt
from simupy.systems import SystemFromCallable, LTISystem
from simupy.utils import array_callable_from_vector_trajectory
from simupy.block_diagram import BlockDiagram
from simupy.matrices import (construct_explicit_matrix, matrix_subs,
                             system_from_matrix_DE)

"""
This example is shows how to solve a quadratic tracking problem.

The plant is

xdot_1(t) = x2(t)
xdot_2(t) = -2*x1(t) -3*x2(t) + u(t)

The goal is to minimize the quadrtic cost

J = integral((x1(t)-2*t)**2 + 0.02*u(t)**2, t, 0, 20)

This cost function implies the tracking problem for x1(t) following r(t) = 2*t.
The optimal controller is found by definining the appropriate differential
algebraic Riccati equation and solve it using the sweep method.

"""

# symbolic matrices
As = construct_explicit_matrix('a', 2, 2, dynamic=False)
Bs = construct_explicit_matrix('b', 2, 1, dynamic=False)
Qs = construct_explicit_matrix('q', 2, 2, symmetric=True, dynamic=False)
Rs = construct_explicit_matrix('r', 1, 1, symmetric=True, dynamic=False)
Ss = construct_explicit_matrix('s', 2, 2, symmetric=True, dynamic=True)
Gs = construct_explicit_matrix('g', 2, 1, dynamic=True)
rxs = construct_explicit_matrix('rx', 2, 1, dynamic=True)

# numerical matrices values
An = np.mat("0,1;-2,-3")
Bn = np.mat("0;1")
Qn = np.mat("1,0;0,0")
Rn = np.mat("0.02")

# matrix differential equations for finite-horizon LQR system
Ssdot = (Qs+As.T*Ss+Ss*As-Ss*Bs*Rs.inv()*Bs.T*Ss)
Gsdot = As.T*Gs - Ss*Bs*Rs.inv()*Bs.T*Gs - Qs*rxs


# combine matrix differential equations
SGdot = Ssdot.row_join(Gsdot)
SG = Ss.row_join(Gs)
SG_subs = dict(matrix_subs((As, An), (Bs, Bn), (Qs, Qn), (Rs, Rn)))

# construct systems from matrix differential equations and reference
SG_sys = system_from_matrix_DE(SGdot, SG,  rxs, SG_subs)


def ref_input_ctr(t, *args):
    return np.r_[2*(tF-t), 0]


ref_input_ctr_sys = SystemFromCallable(ref_input_ctr, 0, 2)

# simulate matrix differential equation with reference input
tF = 20
RiccatiBD = BlockDiagram(SG_sys, ref_input_ctr_sys)
RiccatiBD.connect(ref_input_ctr_sys, SG_sys)
sg_sim_res = RiccatiBD.simulate(tF)


# create callable to interpolate simulation results
sim_data_unique_t, sim_data_unique_t_idx = np.unique(
    sg_sim_res.t,
    return_index=True
)

mat_sg_result = array_callable_from_vector_trajectory(
    np.flipud(tF-sg_sim_res.t[sim_data_unique_t_idx]),
    np.flipud(sg_sim_res.x[sim_data_unique_t_idx]),
    SG_sys.state,
    SG
)
vec_sg_result = array_callable_from_vector_trajectory(
    np.flipud(tF-sg_sim_res.t[sim_data_unique_t_idx]),
    np.flipud(sg_sim_res.x[sim_data_unique_t_idx]),
    SG_sys.state,
    SG_sys.state
)

# Plot S components
plt.figure()
plt.plot()
plt.plot(sg_sim_res.t, mat_sg_result(sg_sim_res.t)[:, [0, 0, 1], [0, 1, 1]])
plt.legend(['$s_{11}$', '$s_{12}$', '$s_{22}$'])
plt.title('unforced component of solution to Riccatti differential equation')
plt.xlabel('$t$, s')
plt.ylabel('$s_{ij}(t)$')
plt.show()

# Plot G components
plt.figure()
plt.plot(sg_sim_res.t, mat_sg_result(sg_sim_res.t)[:, :, -1])
plt.title('forced component of solution to Riccatti differential equation')
plt.legend(['$g_1$', '$g_2$'])
plt.xlabel('$t$, s')
plt.ylabel('$g_{i}(t)$')
plt.show()

# Construct controller from solution to riccati differential algebraic equation
tracking_controller = SystemFromCallable(
    lambda t, xx: -Rn**-1@Bn.T@(mat_sg_result(t)[:, :-1]@xx
                                + mat_sg_result(t)[:, -1]),
    2, 1
)
sys = LTISystem(An, Bn)  # plant system
sys.initial_condition = np.zeros((2, 1))

# simulate controller and plant
control_BD = BlockDiagram(sys, tracking_controller)
control_BD.connect(sys, tracking_controller)
control_BD.connect(tracking_controller, sys)
control_res = control_BD.simulate(tF)

# plot states and reference
plt.figure()
plt.plot(control_res.t, control_res.x, control_res.t, 2*control_res.t)
plt.title('reference and state vs. time')
plt.xlabel('$t$, s')
plt.legend([r'$x_1$', r'$x_2$', r'$r_1$'])
plt.ylabel('$x_{i}(t)$')
plt.show()

# plot control input
plt.figure()
plt.plot(control_res.t, control_res.y[:, -1])
plt.title('optimal control law vs. time')
plt.xlabel('$t$, s')
plt.ylabel('$u(t)$')
plt.show()
