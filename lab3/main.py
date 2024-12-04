# main.py

import numpy as np
import matplotlib.pyplot as plt
from nelder_mead import nelder_mead
from excel import add_to_excel

# Zmienna globalna do śledzenia liczby wywołań funkcji celu
global_fcalls = 0

def funkcja_testowa(x1, x2):
    """
    Funkcja testowa do optymalizacji.
    """
    epsilon = 1e-15
    top = np.sin(np.pi * np.sqrt((x1 / np.pi)**2 + (x2 / np.pi)**2))
    bottom = (np.pi * np.sqrt((x1 / np.pi)**2 + (x2 / np.pi)**2)) + epsilon
    return top / bottom

def funkcja_testowa_external_penalty(x, a, c):
    """
    Funkcja celu z zewnętrzną funkcją kary.
    """
    global global_fcalls
    global_fcalls += 1

    # Obliczenie testowej funkcji celu
    top = np.sin(np.pi * np.sqrt((x[0] / np.pi)**2 + (x[1] / np.pi)**2))
    bottom = np.pi * np.sqrt((x[0] / np.pi)**2 + (x[1] / np.pi)**2)
    result = top / bottom if bottom != 0 else 0  # Unikanie dzielenia przez zero

    # Inicjalizacja funkcji celu
    y = result

    # Obliczenie funkcji kar
    S = 0
    g1 = -x[0] + 1
    g2 = -x[1] + 1
    g3 = np.sqrt(x[0]**2 + x[1]**2) - a

    if g1 > 0:
        S += (max(0, g1))**2
    if g2 > 0:
        S += (max(0, g2))**2
    if g3 > 0:
        S += (max(0, g3))**2

    y += c * S

    # Obliczenie odległości od początku układu współrzędnych
    r = np.linalg.norm(x)

    return y, r

def funkcja_testowa_internal_penalty(x, a, c):
    """
    Funkcja celu z wewnętrzną funkcją kary.
    """
    global global_fcalls
    global_fcalls += 1

    # Obliczenie testowej funkcji celu
    top = np.sin(np.pi * np.sqrt((x[0] / np.pi)**2 + (x[1] / np.pi)**2))
    bottom = np.pi * np.sqrt((x[0] / np.pi)**2 + (x[1] / np.pi)**2)
    result = top / bottom if bottom != 0 else 0  # Unikanie dzielenia przez zero

    # Inicjalizacja funkcji celu
    y = result

    # Obliczenie funkcji kar
    S = 0
    g1 = -x[0] + 1
    g2 = -x[1] + 1
    g3 = np.sqrt(x[0]**2 + x[1]**2) - a

    # Sprawdzenie, czy punkt znajduje się w obszarze dopuszczalnym
    if g1 > 0 or g2 > 0 or g3 > 0:
        return 1e6, np.linalg.norm(x)  # Duża kara za naruszenie ograniczeń

    if g1 < 0:
        S -= 1 / g1
    if g2 < 0:
        S -= 1 / g2
    if g3 < 0:
        S -= 1 / g3

    y += c * S

    # Obliczenie odległości od początku układu współrzędnych
    r = np.linalg.norm(x)

    return y, r

def generate_random_point(a):
    """
    Generuje losowy punkt w obszarze dopuszczalnym.
    """
    while True:
        x1 = np.random.uniform(1, a)
        x2 = np.random.uniform(1, a)
        if np.sqrt(x1**2 + x2**2) <= a:
            return np.array([x1, x2])

if __name__ == '__main__':
    # Parametry
    a_values = [4, 4.4934, 5]
    c = 1000  # Współczynnik kary
    s = 0.1  # Długość boku początkowego sympleksu
    epsilon = 1e-3  # Dokładność optymalizacji
    Nmax = 1000  # Maksymalna liczba wywołań funkcji celu

    line_number = 3  # Numer wiersza w pliku Excel (zaczynamy od 2, zakładając nagłówek)

    for a in a_values:
        # 100 optymalizacji dla każdej wartości a
        for iteration in range(100):
            # Reset liczby wywołań funkcji celu
            global_fcalls = 0

            # Generowanie losowego punktu startowego w obszarze dopuszczalnym
            x0 = generate_random_point(a)

            # Optymalizacja zewnętrzna
            def objective_external(x):
                y, r = funkcja_testowa_external_penalty(x, a, c)
                return y

            res_x_zew, res_f_zew, n_iter_zew = nelder_mead(
                objective_external,
                x0,
                step_size=s,
                tol=epsilon,
                max_iter=Nmax
            )
            n_fcalls_zew = global_fcalls
            y_zew, r_zew = funkcja_testowa_external_penalty(res_x_zew, a, c)

            # Reset liczby wywołań funkcji celu
            global_fcalls = 0

            # Optymalizacja wewnętrzna
            def objective_internal(x):
                y, r = funkcja_testowa_internal_penalty(x, a, c)
                return y

            res_x_wew, res_f_wew, n_iter_wew = nelder_mead(
                objective_internal,
                x0,
                step_size=s,
                tol=epsilon,
                max_iter=Nmax
            )
            n_fcalls_wew = global_fcalls
            y_wew, r_wew = funkcja_testowa_internal_penalty(res_x_wew, a, c)

            # Zapis wyników do pliku Excel
            # add_to_excel(
            #     x0[0], x0[1],
            #     res_x_zew[0], res_x_zew[1], r_zew, y_zew, n_fcalls_zew,
            #     res_x_wew[0], res_x_wew[1], r_wew, y_wew, n_fcalls_wew,
            #     line_number
            # )

            print(f"a: {a}, iteracja: {iteration}, {x0[0], x0[1], res_x_zew[0], res_x_zew[1], r_zew, y_zew, n_fcalls_zew, res_x_wew[0], res_x_wew[1], r_wew, y_wew, n_fcalls_wew, line_number} \n")

            line_number += 1

    print("Optymalizacje zakończone.")
