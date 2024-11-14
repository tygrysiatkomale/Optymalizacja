import numpy as np
from excel import add_to_chart_excel_hooke


def hooke_jeeves(funkcja, punkt_startowy, krok_startowy, alfa, epsilon, maks_wywolan):
    """
    Algorytm optymalizacji Hooke-Jeevesa.

    Parametry:
    - funkcja: Funkcja celu, którą optymalizujemy.
    - punkt_startowy: Punkt początkowy w postaci np. [x1, x2].
    - krok_startowy: Początkowa długość kroku.
    - alfa: Współczynnik zmniejszania kroku, 0 < alfa < 1.
    - epsilon: Dokładność.
    - maks_wywolan: Maksymalna liczba wywołań funkcji celu.

    Zwraca:
    - Punkt optymalny znaleziony przez algorytm.
    """

    def probuj(x, krok):
        for j in range(len(x)):
            if funkcja(x + krok * np.eye(len(x))[j]) < funkcja(x):
                x = x + krok * np.eye(len(x))[j]
            elif funkcja(x - krok * np.eye(len(x))[j]) < funkcja(x):
                x = x - krok * np.eye(len(x))[j]
        return x

    xB = np.array(punkt_startowy) # punktem startowym jest wartosc x
    s = krok_startowy
    liczba_wywolan = 0

    while s > epsilon and liczba_wywolan < maks_wywolan:
        xN = probuj(xB, s)
        if funkcja(xN) < funkcja(xB):
            while funkcja(xN) < funkcja(xB):
                xB = xN
                xN = probuj(xB + (xB - punkt_startowy), s)
                # print(f"fcall: {liczba_wywolan}, x1: {xB[0]}, x2: {xB[1]}")
                # add_to_chart_excel_hooke(xB[0], xB[1], liczba_wywolan+3)
                liczba_wywolan += 1
                if liczba_wywolan >= maks_wywolan:
                    break
        else:
            s *= alfa
            # print(f"fcall: {liczba_wywolan}, x1: {xB[0]}, x2: {xB[1]}")
            # add_to_chart_excel_hooke(xB[0], xB[1], liczba_wywolan + 3)
            liczba_wywolan += 1         # liczba wywolan czyli fcalls

    return xB, liczba_wywolan
