import numpy as np
from simupy.utils import isclose

def test_isclose():
    t1 = np.linspace(0, np.pi/2, num=50)
    t2 = np.linspace(0, np.pi/2, num=30)
    assert isclose(t1, np.sin(t1), t2, np.sin(t2))
    assert isclose(t1, np.sin(t1), t1, np.sin(t1))
    assert not isclose(t1, np.sin(t1), t1, np.cos(t1))

if __name__ == '__main__':
    test_isclose()

