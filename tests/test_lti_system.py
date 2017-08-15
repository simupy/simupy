import numpy as np
import numpy.testing as npt
from simupy.systems import LTISystem
import pytest

def test_k():
    max_m = 4
    max_p = 4
    for m in range(1, max_m+1):
        for p in range(1, max_p+1):
            K = np.random.rand(p,m)
            u = np.random.rand(m)
            sys = LTISystem(K)
            assert sys.dim_state == 0
            assert sys.dim_input == m
            assert sys.dim_output == p
            npt.assert_allclose(
                sys.output_equation_function(p*max_m + m, u),
                K@u
            )

def test_ab():
    max_n = 4
    max_m = 4
    for n in range(1, max_n+1):
        for m in range(1, max_m+1):
            A = np.random.rand(n,n)
            B = np.random.rand(n,m)
            x = np.random.rand(n)
            u = np.random.rand(m)
            if n != m:
                with pytest.raises(AssertionError):
                    LTISystem(np.random.rand(n,m), B)
                with pytest.raises(AssertionError):
                    LTISystem(A, np.random.rand(m,n))

            sys = LTISystem(A, B)
            assert sys.dim_state == n
            assert sys.dim_output == n
            assert sys.dim_input == m

            npt.assert_allclose(
                sys.state_equation_function(n*max_m + m, x, u),
                A@x + B@u
            )
            
            npt.assert_allclose(
                sys.output_equation_function(n*max_m + m, x),
                x
            )

def test_abc():
    max_n = 4
    max_m = 4
    max_p = 4
    for n in range(1, max_n+1):
        for m in range(1, max_m+1):
            for p in range(1, max_p+1):

                A = np.random.rand(n,n)
                B = np.random.rand(n,m)
                C = np.random.rand(p,n)
                x = np.random.rand(n)
                u = np.random.rand(m)
                if p != n:
                    with pytest.raises(AssertionError):
                        LTISystem(A, B, np.random.rand(n, p))

                sys = LTISystem(A, B, C)
                assert sys.dim_state == n
                assert sys.dim_output == p
                assert sys.dim_input == m

                npt.assert_allclose(
                    sys.state_equation_function(n*max_m + m, x, u),
                    A@x + B@u
                )
                
                npt.assert_allclose(
                    sys.output_equation_function(n*max_m + m, x),
                    C@x
                )


