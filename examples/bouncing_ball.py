import numpy as np, matplotlib.pyplot as plt, sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from simupy.systems.symbolic import DynamicalSystem
from simupy.systems import SystemFromCallable
from simupy.discontinuities import SwitchedSystem
from simupy import block_diagram 
from simupy.block_diagram import BlockDiagram
from sympy.tensor.array import Array
from simupy.array import r_,c_

int_opts = block_diagram.DEFAULT_INTEGRATOR_OPTIONS.copy()


plt.ion()

x = x1,x2 = Array(dynamicsymbols('x_1:3'))
mu, g = sp.symbols('mu g')
# constants = {mu: 0.8, g: 9.81}
# sys = SwitchedSystem(
#     x1, Array([0]), state_equations=r_[x2,-g], state_update_equation=r_[sp.Abs(x1),-mu*x2], state=x, constants_values=constants, initial_condition=np.r_[10,15])

constants = {mu: 7/8, g: 7/8}
sys = SwitchedSystem(
    x1, Array([0]), state_equations=r_[x2,-g], state_update_equation=r_[sp.Abs(x1),-mu*x2], state=x, constants_values=constants, initial_condition=np.r_[1,1.125])
    
# test_xs = np.linspace(-3,3)
# res = np.array([sys.event_equation_function(0, test_x, test_x) for test_x in test_xs])
# plt.figure()
# plt.plot(test_xs, res)

BD = BlockDiagram(sys)
# int_opts['rtol'] = 1E-15
# int_opts['atol'] = 1E-15
int_opts['max_step'] = 0.25
res = BD.simulate(25, integrator_options=int_opts)
plt.figure()
plt.plot(res.t,res.x[:,0])