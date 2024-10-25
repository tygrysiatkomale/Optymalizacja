import numpy as np
import matplotlib.pyplot as plt


class FCallsUnique:
    def __init__(self, func):
        self.func = func
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.func(x)


def f(x):
    return -np.cos(0.1 * x) * np.exp(-(0.1 * x - 2 * np.pi) ** 2) + 0.002 * (0.1 * x) ** 2


def make_plot(f, range_start=-100, range_stop=100):
    x = []
    y = []

    for i in range(range_start, range_stop):
        x.append(i)
        y.append(f(i))

    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Wartości')

    plt.title('Wykres funkcji')
    plt.xlabel('Oś X')
    plt.ylabel('Oś Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    return 0


def expansion_method(f, x_0, d, alfa, N_MAX=1000):
    f = FCallsUnique(f)
    i = 0
    x_1 = x_0 + d
    x = [x_0, x_1]

    if f(x[1]) == f(x[0]):
        return x[0], x[1]

    if f(x[1]) > f(x[0]):
        d = -d
        x[1] = x[0] + d
        if f(x[1]) >= f(x[0]):
            return x[1], (x[0] - d)

    while True:
        if f.calls > N_MAX:
            raise ValueError("Error: Przekroczono N_MAX")
        i = i + 1
        x.append(x[0] + (alfa ** i) * d)

        if f(x[i]) <= f(x[i + 1]):
            break

    if d > 0:
        return x[i - 1], x[i + 1]

    return x[i + 1], x[i - 1]


def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[-1] + fib[-2])
    return fib


def metodaFibonacciego(a, b, epsilon):
    n = 1
    fib = fibonacci(100)

    while fib[n] < (b - a) / epsilon:
        n += 1

    k = n
    a_i = a
    b_i = b

    c_i = b_i - (fib[k - 1] / fib[k]) * (b_i - a_i)
    d_i = a_i + b_i - c_i

    for i in range(k - 2):
        if f(c_i) < f(d_i):
            b_i = d_i
            d_i = c_i
            c_i = b_i - (fib[k - i - 2] / fib[k - i - 1]) * (b_i - a_i)
        else:
            a_i = c_i
            c_i = d_i
            d_i = a_i + b_i - c_i

    return (c_i + d_i) / 2


def lagrange_interpolation(func, a, b, c, epsilon=1e-5, gamma=1e-5, max_iter=100):
    i = 0
    while i < max_iter:
        l = (func(a) * (b ** 2 - c ** 2) + func(b) * (c ** 2 - a ** 2) + func(c) * (a ** 2 - b ** 2))
        m = (func(a) * (b - c) + func(b) * (c - a) + func(c) * (a - b))

        if m <= 0:
            raise ValueError("Błąd: podział przez zero lub ujemna wartość w metodzie Lagrange’a.")

        d = 0.5 * l / m

        if a < d < c:
            if func(d) < func(c):
                b = c
                c = d
            else:
                a = d
        elif c < d < b:
            if func(d) < func(c):
                a = c
                c = d
            else:
                b = d
        else:
            raise ValueError("Błąd: punkt interpolacji d jest poza zakresem przedziału.")

        if abs(b - a) < epsilon or abs(d - c) < gamma:
            return d

        i += 1

    raise RuntimeError("Przekroczono maksymalną liczbę iteracji bez zbieżności.")


# Parametry i testy
x0 = 45
d = 1
alpha = 1.5
nmax = 100
epsilon = 0.01

make_plot(f)

expansion_result = expansion_method(f, x0, d, alpha, nmax)
print("Przedział ekspansji: ", expansion_result)

a, b = expansion_result
fib_result = metodaFibonacciego(a, b, epsilon)
print("Przybliżone minimum Fibonacciego: ", fib_result)

# Test metody Lagrange'a
c = (a + b) / 2
lagrange_result = lagrange_interpolation(f, a, b, c, epsilon)
print("Przybliżone minimum Lagrange'a: ", lagrange_result)