from sympy import symbols
from sympy.physics.mechanics import RigidBody as sRigidBody, ReferenceFrame as sReferenceFrame

from sympy.physics.mechanics import inertia, Point
from sympy.physics.vector import Dyadic

# TODO: Point class that automatically applies 1/2-pt A/V theorem to dynamic symbol variables?
# TODO: Add express_as_quaternion translator helpers? 

class ReferenceFrame(sReferenceFrame):
    def __init__(self, *args, **kwargs):
        super(ReferenceFrame,self).__init__(*args,**kwargs)
        self.o = Point(self.name+'o')
        self.o.set_acc(self,0)
        self.o.set_vel(self,0)

class RigidBody(sRigidBody,ReferenceFrame):
    def __init__(self,name,mass,Ixx=None,Iyy=None,Izz=None,Ixz=None,Ixy=None,Iyz=None,indices=None, latexs=None):
        """
        If inertia is not a tuple, assume about own center of mass.

        """
        ReferenceFrame.__init__(self,name,indices,latexs)
        
        
        self.cm = Point(self.name+'cm')
        self.cm.set_acc(self,0)
        self.cm.set_vel(self,0)
            
        sRigidBody.__init__(self, name, self.cm, self, mass,
         (inertia(self, Ixx or 0,Iyy or 0,Izz or 0,Ixz or 0,Ixy or 0,Iyz or 0), self.cm))