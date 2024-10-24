import numpy as np


def f(x):
    return -np.cos(0.1 * x) * np.exp(-(0.1 * x - 2 * np.pi) ** 2) + 0.002 * (0.1 * x) ** 2


def expand_method(f, x0, d, alpha, Nmax):
    i = 0
    fcalls = 0  # Licznik wywołań funkcji celu

    x1 = x0 + d
    fcalls += 1  # Zliczamy wywołanie funkcji dla x1
    if f(x1) == f(x0):
        return [x0, x1]

    if f(x1) < f(x0):
        d = -d
        x1 = x0 + d
        fcalls += 1  # Zliczamy wywołanie funkcji dla nowego x1
        if f(x1) >= f(x0):
            return [x1, x0 - d]

    while True:
        if fcalls > Nmax:
            return "error"

        i += 1
        xi_next = x0 + alpha ** i * d
        fcalls += 1  # Zliczamy wywołanie funkcji dla xi_next

        if f(xi_next) <= f(x1):
            break

    if d > 0:
        return [x1 - d, xi_next]

    return [xi_next, x1 + d]


# Przykładowe dane wejściowe
x0 = 12
d = 1
alpha = 1.5
Nmax = 100

result = expand_method(f, x0, d, alpha, Nmax)
print(result)


print (f"f(0) = {f(0)}, f(10) = {f(10)}, f(66) = {f(66)}, f(67) = {f(67)}, f(68) = {f(68)}")
