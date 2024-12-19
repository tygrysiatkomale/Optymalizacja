import numpy as np

f_calls = 0
g_calls = 0
h_calls = 0


def reset_calls():
    global f_calls, g_calls, h_calls
    f_calls = 0
    g_calls = 0
    h_calls = 0


# Funkcja celu testowa
def ff4T(x):
    global f_calls
    f_calls += 1
    x1, x2 = x
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


# Gradient funkcji celu testowej
def gf4T(x):
    global g_calls
    g_calls += 1
    x1, x2 = x
    df_dx1 = 2 * (x1 + 2 * x2 - 7) + 4 * (2 * x1 + x2 - 5)
    df_dx2 = 4 * (x1 + 2 * x2 - 7) + 2 * (2 * x1 + x2 - 5)
    return np.array([df_dx1, df_dx2])


# Hesjan funkcji celu testowej
def hf4T(x):
    global h_calls
    h_calls += 1
    return np.array([[10, 8], [8, 10]])


# Metoda najszybszego spadku
def steepest_descent(ff, gf, x0, h0, epsilon, Nmax):
    x = np.array(x0)
    for _ in range(Nmax):
        grad = gf(x)
        d = -grad

        # Zmienny krok metodą złotego podziału
        if h0 <= 0:
            h = golden_section_search(lambda h: ff(x + h * d), 0, 1, epsilon, Nmax)
            print(h, h0)
        else:
            h = h0
            print(h, h0)

        x_next = x + h * d
        if np.linalg.norm(x_next - x) < epsilon:
            return x_next
        x = x_next
    return x  # Jeśli nie osiągnięto zbieżności


# Metoda gradientów sprzężonych
def conjugate_gradient(ff, gf, x0, h0, epsilon, Nmax):
    x = np.array(x0)
    g_prev = gf(x)
    d = -g_prev

    for _ in range(Nmax):
        if h0 <= 0:
            h = golden_section_search(lambda h: ff(x + h * d), 0, 1, epsilon, Nmax)
        else:
            h = h0

        x_next = x + h * d
        if np.linalg.norm(x_next - x) < epsilon:
            return x_next

        g_curr = gf(x_next)
        beta = np.dot(g_curr, g_curr) / np.dot(g_prev, g_prev)
        d = -g_curr + beta * d
        g_prev = g_curr
        x = x_next
    return x


# Metoda Newtona
def newton(ff, gf, hf, x0, h0, epsilon, Nmax):
    x = np.array(x0)
    for _ in range(Nmax):
        grad = gf(x)
        hess = hf(x)
        d = -np.linalg.solve(hess, grad)

        if h0 <= 0:
            h = golden_section_search(lambda h: ff(x + h * d), 0, 1, epsilon, Nmax)
        else:
            h = h0

        x_next = x + h * d
        if np.linalg.norm(x_next - x) < epsilon:
            return x_next
        x = x_next
    return x


# Metoda złotego podziału
def golden_section_search(f, a, b, epsilon, Nmax):
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    c = b - resphi * (b - a)
    d = a + resphi * (b - a)

    for _ in range(Nmax):
        if f(c) < f(d):
            b = d
        else:
            a = c

        if abs(b - a) < epsilon:
            return (b + a) / 2

        c = b - resphi * (b - a)
        d = a + resphi * (b - a)
    return (b + a) / 2


# Funkcja aktywacji sigmoid
def sigmoid(theta, x):
    return 1 / (1 + np.exp(-np.dot(theta.T, x)))


# Funkcja kosztu dla problemu rzeczywistego
def ff4R(theta, X, Y):
    m = len(Y)
    return (-1 / m) * np.sum(Y * np.log(sigmoid(theta, X)) + (1 - Y) * np.log(1 - sigmoid(theta, X)))


# Gradient funkcji kosztu dla problemu rzeczywistego
def gf4R(theta, X, Y):
    m = len(Y)
    grad = np.zeros(theta.shape)
    for i in range(m):
        x_i = X[i]
        y_i = Y[i]
        grad += (sigmoid(theta, x_i) - y_i) * x_i
    return grad / m


# Test funkcji na przykładowych danych
if __name__ == "__main__":
    # Parametry początkowe
    x_start = np.random.uniform(-10, 10, 2)
    x0 = [-9.29, 2.4]
    h0 = [0.05, 0.12, 0]
    epsilon = 1e-6
    Nmax = 10000

    # Testowanie algorytmów
    for h in h0:
        for i in range(100):
            x_start = np.random.uniform(-10, 10, 2)
            reset_calls()
            result_sd = steepest_descent(ff4T, gf4T, x0, h, epsilon, Nmax)
            fc = f_calls
            gc = g_calls
            yf = gf4T(result_sd)
            print(f"h: {h}, iter: {i+1}, x1: {x_start[0]}, x2: {x_start[1]}, x1*: {result_sd[0]}, "
                  f"x2*: {result_sd[1]}, "f"y*: {yf}, fcalls: {fc}, gcalls: {gc}")

        print("\n\n gradient \n\n")

        for i in range(100):
            x_start = np.random.uniform(-10, 10, 2)
            reset_calls()
            result_cg = conjugate_gradient(ff4T, gf4T, x0, h0[0], epsilon, Nmax)
            fc = f_calls
            gc = g_calls
            yf = gf4T(result_cg)
            print(f"h: {h}, iter: {i + 1}, x1: {x_start[0]}, x2: {x_start[1]}, x1*: {result_cg[0]}, "
                  f"x2*: {result_cg[1]}, "f"y*: {yf}, fcalls: {fc}, gcalls: {gc}")

        print("\n\n newton \n\n")
        for i in range(100):
            x_start = np.random.uniform(-10, 10, 2)
            reset_calls()
            result_newton = newton(ff4T, gf4T, hf4T, x0, h0[0], epsilon, Nmax)
            fc = f_calls
            gc = g_calls
            yf = gf4T(result_newton)
            print(f"h: {h}, iter: {i + 1}, x1: {x_start[0]}, x2: {x_start[1]}, x1*: {result_newton[0]}, "
                  f"x2*: {result_newton[1]}, "f"y*: {yf}, fcalls: {fc}, gcalls: {gc}")

