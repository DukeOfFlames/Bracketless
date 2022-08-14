import numpy as np
import scipy as sp
import operator
import functools

dx = 1 / 2**6


def poly_for_factorial(m):
    def modified_factorial(x):
        return 1 - sp.special.gamma(x + 1)

    pairs = []

    for x in [0, 1]:
        pairs.append((0, x, 0))  # (n, x, y)

    for i in range(m):
        n_max = 2 * i + 1

        def nth_derivative(n):
            def f(x):
                return sp.misc.derivative(
                    modified_factorial, x, n=n, order=n + (n % 2) + 1, dx=dx
                )

            return f

        root = sp.optimize.root_scalar(
            f=nth_derivative(n_max),
            fprime=nth_derivative(n_max + 1),
            x0=0.5,
            method="newton",
        )
        if not root.converged:
            raise Exception
        x = root.root
        for n in range(n_max + 1):
            y = nth_derivative(n)(x)
            pairs.append((n, x, y))

    A = []
    B = []

    for (n, x, y) in pairs:
        A.append(
            [
                functools.reduce(operator.mul, [exp - i for i in range(n)], 1)
                * x ** (exp - n)
                for exp in range(len(pairs))
            ]
        )  # nth derivative of x**exp
        B.append(y)

    modified_coefficients = np.linalg.solve(np.array(A), np.array(B))
    coefficients = np.array([1] + [0] * (len(pairs) - 1)) - modified_coefficients
    return list(coefficients)


for m in range(4):
    print(poly_for_factorial(m))
