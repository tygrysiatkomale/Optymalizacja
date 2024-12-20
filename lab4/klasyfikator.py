import numpy as np

# Liczniki
f_calls = 0
g_calls = 0

# Reset liczników
def reset_calls():
    global f_calls, g_calls
    f_calls = 0
    g_calls = 0

# Funkcja kosztu
def ff4R(theta, X, Y):
    global f_calls
    f_calls += 1
    m = len(Y)
    h = sigmoid(theta, X)
    return (-1 / m) * np.sum(Y * np.log(h + 1e-15) + (1 - Y) * np.log(1 - h + 1e-15))

# Gradient funkcji kosztu
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

# Funkcja sigmoid
def sigmoid(theta, X):
    z = np.dot(X, theta)
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Metoda gradientów sprzężonych
def conjugate_gradient(ff, gf, x0, h0, epsilon, Nmax, X, Y):
    x = np.array(x0)
    g_prev = gf(x, X, Y)
    d = -g_prev

    for _ in range(Nmax):
        if h0 <= 0:
            h = golden_section_search(lambda h: ff(x + h * d, X, Y), 0, 1, epsilon, Nmax)
        else:
            h = h0

        x_next = x + h * d
        if np.linalg.norm(x_next - x) < epsilon:
            return x_next

        g_curr = gf(x_next, X, Y)
        beta = np.dot(g_curr, g_curr) / np.dot(g_prev, g_prev)
        d = -g_curr + beta * d
        g_prev = g_curr
        x = x_next
        if np.linalg.norm(g_curr) < epsilon:
            return x_next
    return x

# Golden Section Search (placeholder, implement jeśli potrzebne)
def golden_section_search(f, a, b, epsilon, Nmax):
    return 0.1  # Stała wartość kroku, dopasuj do wymagań

# Parametry
theta0 = np.zeros(3)
hs = [0.01, 0.001, 0.0001]
m = 100
epsilon = 1e-6
Nmax = 10000
X = np.hstack((np.ones((m, 1)), np.random.rand(m, 2) * 10))
Y = (X[:, 1] + X[:, 2] > 10).astype(int)

# Tabela wyników
print("Długość kroku\tθ0*\tθ1*\tθ2*\tJ(θ*)\tP(θ*)\tg_calls")
for h in hs:
    reset_calls()
    theta_star = conjugate_gradient(ff4R, gf4R, theta0, h, epsilon, Nmax, X, Y)
    J_star = ff4R(theta_star, X, Y)
    predictions = sigmoid(theta_star, X) >= 0.5
    accuracy = np.mean(predictions == Y) * 100
    print(f"{h:.4f}\t{theta_star[0]:.5f}\t{theta_star[1]:.5f}\t{theta_star[2]:.5f}\t"
          f"{J_star:.5f}\t{accuracy:.2f}\t{g_calls}")
