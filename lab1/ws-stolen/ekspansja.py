from fCalls import *


def metodaEkspansji(f, x_0, d, alfa, N_MAX=1000):
    f = fCallsUnique(f)
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
        if fCalls.calls > N_MAX:
            raise ValueError("Error: Przekroczono N_MAX")
        i = i + 1
        x.append(x[0] + (alfa ** i) * d)

        if f(x[i]) <= f(x[i+1]):
            break

    print(f"Liczba wywołań funkcji: {fCalls.calls}")

    if d > 0:
        return x[i-1], x[i+1]

    return x[i+1], x[i-1]
