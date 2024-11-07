import numpy as np
from scipy.integrate import simps  # Do całkowania numerycznego
from rosenbrock import rosenbrock
from hooke_jeeves import hooke_jeeves


# Funkcja testowa
def funkcja_testowa(x):
    """
    Funkcja testowa do optymalizacji:
    f(x1, x2) = x1^2 + x2^2 - cos(2.5 * pi * x1) - cos(2.5 * pi * x2) + 2
    """
    x1, x2 = x
    return x1 ** 2 + x2 ** 2 - np.cos(2.5 * np.pi * x1) - np.cos(2.5 * np.pi * x2) + 2


# Funkcja kosztu dla ramienia robota
def funkcja_kosztu(k):
    """
    Funkcja kosztu dla optymalizacji współczynników k1 i k2.
    Parametry:
    k: lista zawierająca k1 i k2
    """
    k1, k2 = k
    czas = np.arange(0, 100, 0.1)
    alpha = np.zeros_like(czas)  # kąt
    omega = np.zeros_like(czas)  # prędkość kątowa
    M = np.zeros_like(czas)  # moment siły

    # Moment bezwładności
    I = m1 * l ** 2 + m2 * l ** 2

    # Symulacja ruchu ramienia
    for i in range(1, len(czas)):
        # Moment siły
        M[i] = k1 * (alpha_desired - alpha[i - 1]) + k2 * (omega_desired - omega[i - 1])

        # Równanie ruchu
        omega[i] = omega[i - 1] + (M[i] - b * omega[i - 1]) / I * 0.1
        alpha[i] = alpha[i - 1] + omega[i] * 0.1

    # Funkcja kosztu obliczana metodą prostokątów
    integrand = 10 * (alpha_desired - alpha) ** 2 + (omega_desired - omega) ** 2 + M ** 2
    Q = simps(integrand, czas)  # całkowanie numeryczne

    return Q


# Parametry dla problemu z ramieniem robota
l = 1.0  # długość ramienia
m1 = 1.0  # masa ramienia
m2 = 5.0  # masa ciężarka
b = 0.5  # współczynnik tarcia
alpha_desired = np.pi  # żądany kąt
omega_desired = 0.0  # żądana prędkość kątowa


# Parametry optymalizacji
punkt_startowy = [0.5, 0.5]
krok_startowy = 0.1
alfa = 0.5
epsilon = 1e-6
maks_wywolan = 1000

# Optymalizacja metodą Hooke-Jeevesa
wynik_hooke_jeeves = hooke_jeeves(funkcja_testowa, punkt_startowy, krok_startowy, alfa, epsilon, maks_wywolan)
print("Optymalny punkt (Hooke-Jeeves):", wynik_hooke_jeeves)

# Optymalizacja metodą Rosenbrocka
kroki_startowe = [0.1, 0.1]
wynik_rosenbrock = rosenbrock(funkcja_testowa, punkt_startowy, kroki_startowe, alfa, 0.5, epsilon, maks_wywolan)
print("Optymalny punkt (Rosenbrock):", wynik_rosenbrock)
