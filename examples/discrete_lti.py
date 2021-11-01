import numpy as np
from scipy import signal, linalg
from simupy.systems import LTISystem
from simupy.block_diagram import BlockDiagram
import matplotlib.pyplot as plt

use_model = 1

"""
This example shows several mathematical features of the design of LQR
controllers for a cart-pendulum model (use_model = 0) and a double-integrator
(use_model = 1). This example also shows how a zero-order hold transformation
is exact for LTI systems and how block diagram algebra can be performed to
form equivalent systems.
"""

if use_model == 0:
    m = 1
    M = 3
    L = 0.5
    g = 9.81
    pedant = False
    Ac = np.c_[  # A
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, m*g/M, 0, (-1)**(pedant)*(m+M)*g/(M*L)],
        [0, 0, 1, 0]
    ]
    Bc = np.r_[0, 1/M, 0, 1/(M*L)].reshape(-1, 1)
    Cc = np.eye(4)
    Dc = np.zeros((4, 1))
    ic = np.r_[0, 0, np.pi/3, 0]
elif use_model == 1:
    m = 1
    d = 1
    b = 1
    k = 1
    Ac = np.c_[[0, -k/m], [1, -b/m]]
    Bc = np.r_[0, d/m].reshape(-1, 1)
    Cc = np.eye(2)
    Dc = np.zeros((2, 1))
    ic = np.r_[1, 0]

ct_sys = LTISystem(Ac, Bc, Cc)
ct_sys.initial_condition = ic

n, m = Bc.shape

evals = np.sort(np.abs(np.real(
    linalg.eig(Ac, left=False, right=False, check_finite=False)
)))
dT = 1/(2*evals[-1])
Tsim = (8/np.min(evals[~np.isclose(evals[np.nonzero(evals)], 0)])
        if np.sum(np.isclose(evals[np.nonzero(evals)], 0)) > 0
        else 8
        )

Ad, Bd, Cd, Dd, dT = signal.cont2discrete((Ac, Bc, Cc, Dc), dT)
dt_sys = LTISystem(Ad, Bd, Cd, dt=dT)
dt_sys.initial_condition = ic

Q = np.eye(Ac.shape[0])
R = np.eye(Bc.shape[1] if len(Bc.shape) > 1 else 1)

Sc = linalg.solve_continuous_are(Ac, Bc, Q, R,)
Kc = linalg.solve(R, Bc.T @ Sc).reshape(1, -1)
ct_ctr = LTISystem(-Kc)

Sd = linalg.solve_discrete_are(Ad, Bd, Q, R,)
Kd = linalg.solve(Bd.T @ Sd @ Bd + R, Bd.T @ Sd @ Ad)
dt_ctr = LTISystem(-Kd, dt=dT)


# Equality of discrete-time equivalent and original continuous-time
# system

dtct_bd = BlockDiagram(ct_sys, dt_ctr)
dtct_bd.connect(ct_sys, dt_ctr)
dtct_bd.connect(dt_ctr, ct_sys)
dtct_res = dtct_bd.simulate(Tsim)

dtdt_bd = BlockDiagram(dt_sys, dt_ctr)
dtdt_bd.connect(dt_sys, dt_ctr)
dtdt_bd.connect(dt_ctr, dt_sys)
dtdt_res = dtdt_bd.simulate(Tsim)

plt.figure()
for st in range(n):
    plt.subplot(n+m, 1, st+1)
    plt.plot(dtct_res.t, dtct_res.y[:, st], '+-')
    plt.plot(dtdt_res.t, dtdt_res.y[:, st], 'x-')
    plt.ylabel('$x_{}(t)$'.format(st+1))
    plt.grid(True)
for st in range(m):
    plt.subplot(n+m, 1, st+n+1)
    plt.plot(dtct_res.t, dtct_res.y[:, st+n], '+-')
    plt.plot(dtdt_res.t, dtdt_res.y[:, st+n], 'x-')
    plt.ylabel('$u(t)$')
    plt.grid(True)
plt.xlabel('$t$, s')

plt.subplot(n+m, 1, 1)
plt.title('Equality of discrete-time equivalent and original\n' +
          'continuous-time system subject to same control input')
plt.legend(['continuous-time system', 'discrete-time equivalent'])
# plt.show()

# Equivalence between controlled system and over-all system
ctct_bd = BlockDiagram(ct_sys, ct_ctr)
ctct_bd.connect(ct_sys, ct_ctr)
ctct_bd.connect(ct_ctr, ct_sys)
ctct_res = ctct_bd.simulate(Tsim)

cteq_sys = LTISystem(Ac - Bc @ Kc, np.zeros((n, 0)))
cteq_sys.initial_condition = ic
cteq_res = cteq_sys.simulate(Tsim)

plt.figure()
for st in range(n):
    plt.subplot(n+m, 1, st+1)
    plt.plot(ctct_res.t, ctct_res.y[:, st], '+')
    plt.plot(cteq_res.t, cteq_res.y[:, st], 'x')
    plt.ylabel('$x_{}(t)$'.format(st+1))
    plt.grid(True)
for st in range(m):
    plt.subplot(n+m, 1, st+n+1)
    plt.plot(ctct_res.t, ctct_res.y[:, st+n], '+')
    plt.plot(cteq_res.t, -(Kc@cteq_res.y.T).T, 'x')
    plt.ylabel('$u(t)$')
    plt.grid(True)
plt.xlabel('$t$, s')
plt.subplot(n+m, 1, 1)
plt.title('Equality of system under feedback control and\n' +
          'equivalent closed-loop, continuous time')
plt.legend([r'$\dot{x}(t) = A\ x(t) + B\ u(t)$; $u(t) = K\ x(t)$',
            r'$\dot{x}(t) = (A - B\ K)\ x(t)$'])
# plt.show()

dteq_sys = LTISystem(Ad - Bd @ Kd, np.zeros((n, 0)), dt=dT)
dteq_sys.initial_condition = ic
dteq_res = dteq_sys.simulate(Tsim)

plt.figure()
for st in range(n):
    plt.subplot(n+m, 1, st+1)
    plt.plot(dtdt_res.t, dtdt_res.y[:, st],)
    plt.plot(dteq_res.t, dteq_res.y[:, st],)
    plt.grid(True)
    plt.ylabel('$x_{}(t)$'.format(st+1))
for st in range(m):
    plt.subplot(n+m, 1, st+n+1)
    plt.plot(dtdt_res.t, dtdt_res.y[:, st+n],)
    plt.plot(dteq_res.t, -(Kd@dteq_res.y.T).T,)
    plt.grid(True)
    plt.ylabel('$u(t)$')
plt.xlabel('$t$, s')
plt.subplot(n+m, 1, 1)
plt.title('Equality of system under feedback control and\n' +
          'equivalent closed-loop, discrete time')
plt.legend([r'$x[k+1] = A\ x[k] + B\ u[k]$; $u[k] = K\ x[k]$',
            r'$x[k+1] = (A - B\ K)\ x[k]$'])
plt.show()
