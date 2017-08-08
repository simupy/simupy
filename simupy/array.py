from sympy.tensor.array import Array
from numpy.lib.index_tricks import RClass, CClass


class SymAxisConcatenatorMixin:
    """
    A mix-in to convert numpy AxisConcatenator classes to use with sympy N-D
    arrays.
    """
    def __getitem__(self, key):
        return Array(super().__getitem__(tuple(
            Array(k) if hasattr(k, '__len__') else Array([k]) for k in key
        )))


class SymRClass(SymAxisConcatenatorMixin, RClass):
    pass


class SymCClass(SymAxisConcatenatorMixin, CClass):
    pass


r_ = SymRClass()
c_ = SymCClass()


def empty_array():
    """
    Construct an empty array, which is often needed as a place-holder
    """
    a = Array([0])
    a._shape = tuple()
    a._rank = 0
    a._loop_size = 0
    a._array = []
    a.__str__ = lambda *args, **kwargs: "[]"
    return a
