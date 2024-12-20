import numpy as np
from openpyxl import Workbook, load_workbook

f_calls = 0
g_calls = 0
h_calls = 0


def add_to_excel(start, amount, tab):
    file_name = "xlsx4.xlsx"
    workbook = load_workbook(file_name)
    sheet = workbook["Tabela 1"]
    i = start
    for element in tab:
        sheet[f'C{i}'] = element[0]
        sheet[f'D{i}'] = element[1]
        sheet[f'E{i}'] = element[2]
        sheet[f'F{i}'] = element[3]
        sheet[f'G{i}'] = element[4]
        sheet[f'H{i}'] = element[5]
        sheet[f'I{i}'] = element[6]
        sheet[f'J{i}'] = element[7]
        sheet[f'K{i}'] = element[8]
        sheet[f'L{i}'] = element[9]
        sheet[f'M{i}'] = element[10]
        sheet[f'N{i}'] = element[11]
        sheet[f'O{i}'] = element[12]
        sheet[f'P{i}'] = element[13]
        sheet[f'Q{i}'] = element[14]
        sheet[f'R{i}'] = element[15]
        sheet[f'S{i}'] = element[16]
        sheet[f'T{i}'] = element[17]
        i += 1
    workbook.save(file_name)


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
        else:
            h = h0

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
    global g_calls
    g_calls += 1
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
    h = 0
    epsilon = 1e-6
    Nmax = 10000
    results = []
    # Testowanie algorytmów
    # for h in h0:
    # for i in range(100):
    #     x_start = np.random.uniform(-10, 10, 2)
    #     reset_calls()
    #     result_sd = steepest_descent(ff4T, gf4T, x_start, h, epsilon, Nmax)
    #     y_sd = ff4T(result_sd)
    #     fcalls_sd = f_calls
    #     gcalls_sd = g_calls
    #     print(f"h: {h}, iter: {i+1}, x1: {x_start[0]}, x2: {x_start[1]}, x1*: {result_sd[0]}, "
    #           f"x2*: {result_sd[1]}, "f"y*: {y_sd}, fcalls: {fcalls_sd}, gcalls: {gcalls_sd}")
    #
    #     reset_calls()
    #     result_cg = conjugate_gradient(ff4T, gf4T, x_start, h, epsilon, Nmax)
    #     y_cg = ff4T(result_cg)
    #     fcalls_cg = f_calls
    #     gcalls_cg = g_calls
    #
    #     print(f"h: {h}, iter: {i + 1}, x1: {x_start[0]}, x2: {x_start[1]}, x1*: {result_cg[0]}, "
    #           f"x2*: {result_cg[1]}, "f"y*: {y_cg}, fcalls: {fcalls_cg}, gcalls: {gcalls_cg}")
    #
    #     reset_calls()
    #     result_newton = newton(ff4T, gf4T, hf4T, x_start, h, epsilon, Nmax)
    #     y_newton = ff4T(result_newton)
    #     fcalls_newton = f_calls
    #     gcalls_newton = g_calls
    #     hcalls_newton = h_calls
    #     print(f"h: {h}, iter: {i + 1}, x1: {x_start[0]}, x2: {x_start[1]}, x1*: {result_newton[0]}, "
    #           f"x2*: {result_newton[1]}, "f"y*: {y_newton}, fcalls: {fcalls_newton}, gcalls: {gcalls_newton}, "
    #           f"hcalls: {hcalls_newton}")
    #
    #     results.append([x_start[0], x_start[1],
    #                     result_sd[0], result_sd[1], y_sd, fcalls_sd, gcalls_sd,
    #                     result_cg[0], result_cg[1], y_cg, fcalls_cg, gcalls_cg,
    #                     result_newton[0], result_newton[1], y_newton, fcalls_newton, gcalls_newton, hcalls_newton])
    #
    # add_to_excel(203, 100, results)
    # print(len(results))

    theta0 = np.zeros(3)
    hs = [0.01, 0.001, 0.0001]
    m = 100  # Liczba przypadków
    X = np.hstack((np.ones((m, 1)), np.random.rand(m, 2) * 10))  # Pierwsza kolumna: 1
    Y = (X[:, 1] + X[:, 2] > 10).astype(int)  # Przykładowa klasyfikacja

    # Tabela wyników
    print("Długość kroku\tθ0*\tθ1*\tθ2*\tJ(θ*)\tP(θ*)\tg_calls")
    for h in hs:
        # Reset liczników
        reset_calls()

        # Optymalizacja
        theta_star = conjugate_gradient(ff4R, gf4R, theta0, h, epsilon, Nmax)

        # Wyniki
        J_star = ff4R(theta_star, X, Y)  # Funkcja kosztu w punkcie optymalnym
        predictions = sigmoid(theta_star, X) >= 0.5  # Hipoteza (klasyfikacja)
        accuracy = np.mean(predictions == Y) * 100  # Dokładność
        print(f"{h:.4f}\t{theta_star[0]:.5f}\t{theta_star[1]:.5f}\t{theta_star[2]:.5f}\t"
              f"{J_star:.5f}\t{accuracy:.2f}\t{g_calls}")

