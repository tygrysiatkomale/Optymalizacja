import numpy as np
import pandas as pd
from scipy.integrate import simps  # Do całkowania numerycznego
from rosenbrock import rosenbrock, wyniki_iteracji
from hooke_jeeves import hooke_jeeves

# # funkcja sluzaca do sprawdzania poprawnosci dzialania metod, znamy jej minima
# def funkcja_t(x):
#     x1, x2 = x
#     return (x1 + 3) ** 2 + (x2 -2) ** 2

# Funkcja testowa
def funkcja_testowa(x):
    """
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
# punkt_startowy = [0.5, 0.5]       #ten pkt startowy sluzyl sprawdzeniu poprawnosci dzialania metod

alfa_hooke = 0.5        # wieksza od 0, mniejsza od 1
alfa_rosen = 1.5        # wieksza od 1
beta = 0.5          # wieksza od 0, mniejsza od 1
epsilon = 1e-6
maks_wywolan = 100

krok_startowy = 0.1     # dla hooke_jeeves jest ta zmienna
kroki_startowe = [0.1, 0.1]      # dla rosenbrocka jest ta zmienna


# Losowanie punktu początkowego z przedziału [-1, 1]
punkt_startowy = np.random.uniform(-1, 1, size=2)

'''
To co znajduje się poniżej  ma się znaleźć w excelu w Tabeli 1,
Wykonujemy 100 optymalizacji (na razie nie ma pętli, żeby weryfikować wyniki)
 dla 3 roznych dlugosci kroku i startujemy z losowego punktu.
Wyświetlam nasze wylosowane punkty startowe (kolumna C i D w tabeli)
Wartosc kroku startowego wynosi 0.1, można też dać dla wynoszacego 0.2, 0.3 (skonsultuje z kimś)
~wywołanie algorytmu Hooke-Jeevesa~
wynik_hooke_jeeves -> zwraca nam tablicę z dwoma elementami (x1,x2), które (jak dobrze rozumiem) są minimami XD
liczba_wywolan -> liczba wywołań funkcji celu,
y_hooke_jeeves -> to wartość funkcji celu w znalezionym punkcie (czat mi zasugerowal, ze to jest wlasnie tym y* w tabeli,
   ale zostawiam to do konsultacji z innymi)
zostaje jeszcze określenie czy TAK/NIE minimum globalne, ale mi sie coś sypnelo, wiec na razie nie ma
~ analogicznie jest dla wywołania algorytmu Rosenbrocka ~
różnica:
wartość kroków startowych może wynosić [0.2, 0.2], [0.3, 0.3] (nie pomylic ze tu jest tablica z dwoma wartosciami)

'''

wynik_hooke_jeeves, liczba_wywolan = hooke_jeeves(funkcja_testowa, punkt_startowy, krok_startowy, alfa_hooke, epsilon, maks_wywolan)
y_hooke_jeeves = funkcja_testowa(wynik_hooke_jeeves)

print("== Wyniki optymalizacji metodą Hooke-Jeevesa ==")
print(f"Punkt startowy: x1 = {punkt_startowy[0]}, x2 = {punkt_startowy[1]}")
print(f"Znaleziony punkt optymalny: x1* = {wynik_hooke_jeeves[0]} i x2* = {wynik_hooke_jeeves[1]}")
print(f"Liczba wywolan funkcji celu: {liczba_wywolan}")
print(f"Wartość funkcji celu w punkcie (y*): {y_hooke_jeeves}")
print("")

wynik_rosenbrock, liczba_wywolan = rosenbrock(funkcja_testowa, punkt_startowy, kroki_startowe, alfa_rosen, beta, epsilon, maks_wywolan)
y_rosenbrock = funkcja_testowa(wynik_rosenbrock)

print("== Wyniki optymalizacji metodą Rosenbrocka ==")
print(f"Punkt startowy: x1 = {punkt_startowy[0]}, x2 = {punkt_startowy[1]}")
print(f"Znaleziony punkt optymalny: x1* = {wynik_rosenbrock[0]:}, x2* = {wynik_rosenbrock[1]:}")
print(f"Liczba wywolan funkcji celu: {liczba_wywolan}")
print(f"Wartość funkcji celu w punkcie (y*): {y_rosenbrock:}")
