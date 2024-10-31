# plik z algorytmami - ekspansja, lagrange, fibonacci
import numpy as np


class FCallsUnique:
    def __init__(self, func):
        self.func = func
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.func(x)

# metoda ekspansji
def expansion_method(f, x_0, d, alfa, n_max=1000):
    f = FCallsUnique(f)
    i = 0
    x_1 = x_0 + d
    x = [x_0, x_1]

    if f(x[1]) == f(x[0]):
        return x[0], x[1], f.calls

    if f(x[1]) > f(x[0]):
        d = -d
        x[1] = x[0] + d
        if f(x[1]) >= f(x[0]):
            return x[1], (x[0] - d), f.calls

    while True:
        if f.calls > n_max:
            raise ValueError("Error: Przekroczono N_MAX")
        i = i + 1
        x.append(x[0] + (alfa ** i) * d)

        if f(x[i]) <= f(x[i + 1]):
            break

    if d > 0:
        return x[i - 1], x[i + 1], f.calls

    return x[i + 1], x[i - 1], f.calls

# metoda fibonacciego
def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[-1] + fib[-2])
    return fib


def fibonacci_method(f, a, b, epsilon):
    f = FCallsUnique(f)
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

    return (c_i + d_i) / 2, f.calls


# metoda lagrange (z poprawkami stabilności)
def lagrange_interpolation(f, a, b, c, epsilon=0.00001, gamma=0.00001, max_iter=10000):
    f = FCallsUnique(f)
    i = 0
    min_m_threshold = 1e-14  # Minimalna wartość graniczna dla mianownika, aby uniknąć dzielenia przez małe wartości

    while i < max_iter:
        l = (f(a) * (b ** 2 - c ** 2) + f(b) * (c ** 2 - a ** 2) + f(c) * (a ** 2 - b ** 2))
        m = (f(a) * (b - c) + f(b) * (c - a) + f(c) * (a - b))

        if abs(m) < min_m_threshold:
            raise ValueError("Błąd: wartość mianownika m jest zbyt bliska zeru, co prowadzi do niestabilności w metodzie Lagrange’a.")

        d = 0.5 * l / m

        # Upewnij się, że d jest w odpowiednim przedziale [a, b]
        if not (a < d < b):
            raise ValueError("Błąd: punkt interpolacji d jest poza zakresem przedziału.")

        if a < d < c:
            if f(d) < f(c):
                b = c
                c = d
            else:
                a = d
        elif c < d < b:
            if f(d) < f(c):
                a = c
                c = d
            else:
                b = d
        else:
            raise ValueError("Błąd: punkt interpolacji d jest poza zakresem przedziału.")

        # Sprawdzenie kryterium zbieżności
        if abs(b - a) < epsilon or abs(d - c) < gamma:
            return d, f.calls

        i += 1

    raise RuntimeError("Przekroczono maksymalną liczbę iteracji bez zbieżności.")