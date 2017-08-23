import numpy as np, matplotlib.pyplot as plt, sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from simupy.systems.symbolic import DynamicalSystem
from simupy.systems import SystemFromCallable
from simupy.discontinuities import SwitchedSystem
from simupy import block_diagram 
from simupy.block_diagram import BlockDiagram
from sympy.tensor.array import Array
from simupy.array import r_,c_
from simupy.utils import callable_from_trajectory

int_opts = block_diagram.DEFAULT_INTEGRATOR_OPTIONS.copy()


plt.ion()

x = x1,x2 = Array(dynamicsymbols('x_1:3'))
mu, g = sp.symbols('mu g')

constants = {mu: 7/8, g: 7/8}
ic = np.r_[1, 1.125]


constants = {mu: 0.8, g: 9.81}
ic = np.r_[10, 15]
sys = SwitchedSystem(
    x1, Array([0]), state_equations=r_[x2,-g], state_update_equation=r_[sp.Abs(x1),-mu*x2], state=x, constants_values=constants, initial_condition=ic)
    
# test_xs = np.linspace(-3,3)
# res = np.array([sys.event_equation_function(0, test_x, test_x) for test_x in test_xs])
# plt.figure()
# plt.plot(test_xs, res)

BD = BlockDiagram(sys)
int_opts['rtol'] = 1E-12
int_opts['atol'] = 1E-15
int_opts['nsteps'] = 1000
# int_opts['max_step'] = 2**-3
res = BD.simulate(np.arange(0,25,2**-8), 'dopri5', integrator_options=int_opts)
unique_t, unique_t_sel = np.unique(res.t, return_index=True)
# clbl = callable_from_trajectory(unique_t, res.x[unique_t_sel,:])

t_sel = res.t > 20

plt.figure()
plt.subplot(2,1,1)
plt.plot(res.t[t_sel],res.x[t_sel,0])
# plt.plot(np.linspace(0,6, 101), clbl(np.linspace(0,6, 101))[:, 0])
plt.subplot(2,1,2)
plt.plot(res.t[t_sel],res.x[t_sel,1])
# plt.plot(np.linspace(0,6), clbl(np.linspace(0,6))[:, 1])

