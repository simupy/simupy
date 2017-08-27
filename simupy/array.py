from sympy.tensor.array import Array
from sympy import ImmutableDenseMatrix as Matrix
from numpy.lib.index_tricks import RClass, CClass, AxisConcatenator


class SymAxisConcatenatorMixin:
    """
    A mix-in to convert numpy AxisConcatenator classes to use with sympy N-D
    arrays.
    """

    # support numpy >= 1.13
    concatenate = staticmethod(
        lambda *args, **kwargs: Array(
            AxisConcatenator.concatenate(*args, **kwargs)
        )
    )
    makemat = staticmethod(Matrix)

    def _retval(self, res):  # support numpy < 1.13
        if self.matrix:
            cls = Matrix
        else:
            cls = Array
        return cls(super()._retval(res))

    def __getitem__(self, key):
        return super().__getitem__(tuple(
            k if isinstance(k, str) else
            Array(k) if hasattr(k, '__len__')
            else Array([k])
            for k in key
        ))


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
