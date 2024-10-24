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

    # print(f"Liczba wywołań funkcji: {f.calls}")

    if d > 0:
        return x[i - 1], x[i + 1]

    return x[i + 1], x[i - 1]


def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n+1):
        fib.append(fib[-1] + fib[-2])
    return fib


def metodaFibonacciego(a, b, epsilon):
    # Krok 1: Znalezienie najmniejszego k, dla którego φk > (b - a) / ε
    n = 1
    fib = fibonacci(100)  # Generujemy liczby Fibonacciego (potrzebujemy tylko do k)

    while fib[n] < (b - a) / epsilon:
        n += 1

    # Krok 2: Początkowe wartości
    k = n
    a_i = a
    b_i = b

    # Krok 3 i 4: Początkowe wartości c(0) i d(0)
    c_i = b_i - (fib[k - 1] / fib[k]) * (b_i - a_i)
    d_i = a_i + b_i - c_i

    # Krok 5: Iteracje
    for i in range(k - 2):
        if f(c_i) < f(d_i):
            b_i = d_i
            d_i = c_i
            c_i = b_i - (fib[k - i - 2] / fib[k - i - 1]) * (b_i - a_i)
        else:
            a_i = c_i
            c_i = d_i
            d_i = a_i + b_i - c_i

    # Krok 16: Zwrócenie wyniku
    return (c_i + d_i) / 2


# Parametry
x0 = 45
d = 1
alpha = 1.5
nmax = 100
epsilon = 0.01

make_plot(f)

expansion_result = expansion_method(f, x0, d, alpha, nmax)
print("Przedział ekspansji: ", expansion_result)

    # fibonnaci
a, b = expansion_result

fib_result = metodaFibonacciego(a, b, epsilon)
print("Przybliżone minimum Fibonacciego: ", fib_result)



