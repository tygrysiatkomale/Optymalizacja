

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
def lagrange_interpolation(f, a, b, c, epsilon=1e-5, max_iter=10000):
    f = FCallsUnique(f)
    i = 0

    for _ in range(max_iter):
        # Sortuj punkty według ich wartości x
        x = sorted([a, b, c])
        x0, x1, x2 = x
        f0, f1, f2 = f(x0), f(x1), f(x2)

        # Obliczanie współczynników interpolacji kwadratowej
        numerator = ((x1 - x0)**2 * (f1 - f2) - (x1 - x2)**2 * (f1 - f0))
        denominator = ((x1 - x0)*(f1 - f2) - (x1 - x2)*(f1 - f0))

        if denominator == 0:
            raise ZeroDivisionError("Dzielenie przez zero w obliczaniu punktu interpolacji.")

        d = x1 - 0.5 * numerator / denominator

        # Sprawdzenie kryterium zbieżności
        if abs(d - x1) < epsilon:
            return d, f.calls

        # Aktualizacja punktów
        fd = f(d)
        if fd < f1:
            if d < x1:
                x2 = x1
                x1 = d
            else:
                x0 = x1
                x1 = d
        else:
            if d < x1:
                x0 = d
            else:
                x2 = d

        a, b, c = x0, x1, x2
        i += 1

    raise RuntimeError("Przekroczono maksymalną liczbę iteracji bez zbieżności.")
