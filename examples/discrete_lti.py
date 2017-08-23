import numpy as np
import scipy.signal as signal
import scipy.linalg as linalg
from simupy.systems import LTISystem, SystemFromCallable
from simupy.block_diagram import BlockDiagram
import control
import matplotlib.pyplot as plt

T=1
K=0.692/(T**2/2)

num = np.r_[1.]
den = np.r_[1.,0,0]
dt0 = signal.TransferFunction(num,den).to_ss().to_discrete(T)
Ac,Bc,Cc,Dc = signal.tf2ss(num,den)
dt1 = Ad, Bd, Cd, Dd, dT = signal.cont2discrete((Ac,Bc,Cc,Dc),T)


# 
# numd, dend, ddt = signal.cont2discrete((num,den),T)
# dt2 = signal.TransferFunction(np.r_[0,1,1]*T**2, np.r_[1, -2, 1]*2, dt=1).to_ss()
# dt3 = signal.TransferFunction(numd, dend, dt=ddt).to_ss()


ctrl_dlti = signal.TransferFunction(K*np.r_[1, -0.63], np.r_[1, 0.44], dt=T)

Q = np.eye(2)
R = np.c_[1]
S = linalg.solve_discrete_are(Ad, Bd, Q, R,)
Kd = linalg.solve(Bd.T @ S @ Bd + R, Bd.T @ S @ Ad)

# sysd = LTISystem(dt3.A, dt3.B, dt3.C, dt=T)
# sysd = LTISystem(dt1[0], dt1[1], dt1[2], dt=T)
sysd = LTISystem(Ad, Bd, dt=T)
sysc = LTISystem(Ac, Bc,)
ctrl = LTISystem(-Kd, dt=T)

bd = BlockDiagram(sysc, sysd, ctrl)
bd = BlockDiagram( sysd, ctrl)
sysc.initial_condition = np.r_[-1,0]
sysd.initial_condition = np.r_[-1,0]
bd.connect(sysd, ctrl)
bd.connect(ctrl, sysd)
# bd.connect(ctrl, sysc)

res0 = bd.simulate(np.arange(0,16, 0.5))
res1 = bd.simulate(np.arange(0,16))
res = bd.simulate(15)

plt.ion()
plt.figure()
plt.subplot(3,1,1)

# plt.plot(res.t, res.y[:,0])
# plt.plot(res0.t, res0.y[:,0])
plt.plot(res.t, res.x[:,0], 'o')
plt.plot(res1.t, res1.x[:,0], '+')
plt.plot(res0.t, res0.x[:,0], 'x')

plt.subplot(3,1,2)

# plt.plot(res.t, res.y[:,0])
# plt.plot(res0.t, res0.y[:,0])
plt.plot(res.t, res.y[:,0], 'o')
plt.plot(res1.t, res1.y[:,0], '+')
plt.plot(res0.t, res0.y[:,0], 'x')

# plt.plot(res0.t, res0.y[:,2], 'o')
# plt.plot(res.t, res.y[:,2], 'x')


plt.subplot(3,1,3)
plt.plot(res.t, res.y[:,-1], 'o')
plt.plot(res1.t, res1.y[:,-1], '+')
plt.plot(res0.t, res0.y[:,-1], 'x')


"""
##
comp1 = LTISystem(ctrl_dlti.A, ctrl_dlti.B, ctrl_dlti.B, dt=T)
comp2 = LTISystem( np.c_[-ctrl_dlti.D, 1], dt=T)


bd = BlockDiagram(sysc, sysd, comp1, comp2)
sysc.initial_condition = np.r_[0,-1]
sysd.initial_condition = np.r_[0,-1]
# bd.connect(sys, err)
bd.connect(sysc, comp1)
bd.connect(sysc, comp2, inputs=[0])
bd.connect(comp1, comp2, inputs=[1])
bd.connect(comp2, sysc)
bd.connect(comp2, sysd)
res0 = bd.simulate(np.arange(16))
res = bd.simulate(15)

plt.ion()
plt.subplot(2,1,1)
plt.plot(res.t, res.y[:,0])
plt.plot(res0.t, res0.y[:,0])
plt.step(res.t, res.y[:,1],)
plt.plot(res0.t, res0.y[:,1], 'x')
plt.subplot(2,1,2)
plt.step(res.t, res.y[:,-1])
plt.plot(res0.t, res0.y[:,-1], 'x')
"""