import numpy as np

# globalna lista na wyniki operacji
wyniki_iteracji = []

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
    # Lista do przechowywania wyników iteracji  -> potrzebne do arkusza 'Wykres' w excelu
    global wyniki_iteracji

    xB = np.array(punkt_startowy)
    s = np.array(kroki_startowe)
    liczba_wywolan = 0

    # Tworzymy początkową bazę kierunków - macierz jednostkowa
    n = len(xB)
    d = np.eye(n)
    lamda = np.zeros(n)
    p = np.zeros(n)

    i = 0
    while np.max(np.abs(s)) > epsilon and liczba_wywolan < maks_wywolan:
        # Zapisujemy aktualny stan w iteracji
        # wyniki_iteracji.append((i, *xB))  # Zapisujemy numer iteracji i współrzędne
        # -> potrzebne do arkusza 'Wykres' w excelu NIE DZIALA

        for j in range(n):
            x_temp = xB + s[j] * d[j]
            if funkcja(x_temp) < funkcja(xB):
                xB = x_temp
                lamda[j] += s[j]
                s[j] *= alfa
            else:
                p[j] += 1
                s[j] *= -beta
            liczba_wywolan += 1
            if liczba_wywolan >= maks_wywolan:
                break

        # Sprawdzamy, czy wszystkie kroki pogorszyły wynik
        if all(p > 0) and all(lamda != 0):
            Q = d * lamda[:, None]
            for i in range(n):
                v = Q[i]
                for j in range(i):
                    v -= np.dot(Q[i], d[j]) * d[j]
                d[i] = v / np.linalg.norm(v)

            s = np.array(kroki_startowe)
            lamda = np.zeros(n)
            p = np.zeros(n)

        i += 1  # Inkrementacja liczby iteracji

    # Zapisujemy wynik końcowy
    # wyniki_iteracji.append((i, *xB))

    return xB
