import numpy as np

def f(x):
    return -np.cos(0.1 * x) * np.exp(-(0.1 * x - 2 * np.pi)**2) + 0.002 * (0.1 * x) ** 2


def expand_method(x0, d, alpha, Nmax):
    return 0


x0 = 1.0
d = 1
alpha = 1.5
Nmax = 100

result = expand_method(x0, d, alpha, Nmax)
print(result)


