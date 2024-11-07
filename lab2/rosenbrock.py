import numpy as np


def rosenbrock(funkcja, punkt_startowy, kroki_startowe, alfa, beta, epsilon, maks_wywolan):
    """
    Algorytm optymalizacji Rosenbrocka.

    Parametry:
    - funkcja: Funkcja celu, którą optymalizujemy.
    - punkt_startowy: Punkt początkowy w postaci np. [x1, x2].
    - kroki_startowe: Początkowa długość kroków dla każdego kierunku.
    - alfa: Współczynnik ekspansji, alfa > 1.
    - beta: Współczynnik kontrakcji, 0 < beta < 1.
    - epsilon: Dokładność.
    - maks_wywolan: Maksymalna liczba wywołań funkcji celu.

    Zwraca:
    - Punkt optymalny znaleziony przez algorytm.
    """
    xB = np.array(punkt_startowy)
    s = np.array(kroki_startowe)
    liczba_wywolan = 0

    i = 0
    while np.max(np.abs(s)) > epsilon and liczba_wywolan < maks_wywolan:
        for j in range(len(xB)):
            if funkcja(xB + s[j] * np.eye(len(xB))[j]) < funkcja(xB):
                xB = xB + s[j] * np.eye(len(xB))[j]
                s[j] *= alfa
            else:
                s[j] *= -beta
            liczba_wywolan += 1
            if liczba_wywolan >= maks_wywolan:
                break
        i += 1

    return xB
