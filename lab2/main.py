import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from rosenbrock import rosenbrock, wyniki_iteracji
from hooke_jeeves import hooke_jeeves
from excel import add_to_excel


def funkcja_testowa(x):
    """
    f(x1, x2) = x1^2 + x2^2 - cos(2.5 * pi * x1) - cos(2.5 * pi * x2) + 2
    """
    x1, x2 = x
    return x1 ** 2 + x2 ** 2 - np.cos(2.5 * np.pi * x1) - np.cos(2.5 * np.pi * x2) + 2


def funkcja_celu(k, return_full=False):
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
    Q = simpson(integrand, x=czas)  # całkowanie numeryczne

    if return_full:
        return czas, alpha, omega
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
maks_wywolan = 1000

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
punkt_startowy = np.random.uniform(-1, 1, size=2)
krok_startowy = 0.1     # dla hooke_jeeves jest ta zmienna
kroki_startowe = [0.1, 0.1]      # dla rosenbrocka jest ta zmienna

"""
Dodawanie do excela ponizej ktore trwa mega dlugo 
"""
# for i in range(3, 303):
#     if (i < 103) and (i > 0):
#         krok_startowy = 0.1
#         kroki_startowe = [0.1, 0.1]
#     elif (i >= 103) and (i < 203):
#         krok_startowy = 0.3
#         kroki_startowe = [0.3, 0.3]
#     elif (i >= 203) and (i < 303):
#         krok_startowy = 0.5
#         kroki_startowe = [0.5, 0.5]
#
#     # Losowanie punktu początkowego z przedziału [-1, 1]
#     punkt_startowy = np.random.uniform(-1, 1, size=2)
#
#     wynik_hooke_jeeves, liczba_wywolan_h = hooke_jeeves(funkcja_testowa, punkt_startowy, krok_startowy, alfa_hooke,
#                                                         epsilon, maks_wywolan)
#     y_hooke_jeeves = funkcja_testowa(wynik_hooke_jeeves)
#
#     wynik_rosenbrock, liczba_wywolan_r = rosenbrock(funkcja_testowa, punkt_startowy, kroki_startowe, alfa_rosen, beta,
#                                                     epsilon, maks_wywolan)
#     y_rosenbrock = funkcja_testowa(wynik_rosenbrock)
#
#     add_to_excel(punkt_startowy[0], punkt_startowy[1],
#                  wynik_hooke_jeeves[0], wynik_hooke_jeeves[1], y_hooke_jeeves, liczba_wywolan_h,
#                  wynik_rosenbrock[0], wynik_rosenbrock[1], y_rosenbrock, liczba_wywolan_r, i)

"""
Tu się kończy dodawanie do excela ^^
"""

wynik_hooke_jeeves, liczba_wywolan_h = hooke_jeeves(funkcja_testowa, punkt_startowy, krok_startowy, alfa_hooke, epsilon, maks_wywolan)
y_hooke_jeeves = funkcja_testowa(wynik_hooke_jeeves)

wynik_rosenbrock, liczba_wywolan_r = rosenbrock(funkcja_testowa, punkt_startowy, kroki_startowe, alfa_rosen, beta, epsilon, maks_wywolan)
y_rosenbrock = funkcja_testowa(wynik_rosenbrock)

print("Optymalizacja dla testowej funkcji:")
print("Wyniki optymalizacji metodą Hooke-Jeevesa: ")
print(f"Punkt startowy: x1 = {punkt_startowy[0]}, x2 = {punkt_startowy[1]}")
print(f"Znaleziony punkt optymalny: x1* = {wynik_hooke_jeeves[0]} i x2* = {wynik_hooke_jeeves[1]}")
print(f"Liczba wywolan funkcji celu: {liczba_wywolan_h}")
print(f"Wartość funkcji celu w punkcie (y*): {y_hooke_jeeves}")

print("Wyniki optymalizacji metodą Rosenbrocka: ")
print(f"Punkt startowy: x1 = {punkt_startowy[0]}, x2 = {punkt_startowy[1]}")
print(f"Znaleziony punkt optymalny: x1* = {wynik_rosenbrock[0]:}, x2* = {wynik_rosenbrock[1]:}")
print(f"Liczba wywolan funkcji celu: {liczba_wywolan_r}")
print(f"Wartość funkcji celu w punkcie (y*): {y_rosenbrock:}")
print("")

'''
Tabela 2
 Trzeba wpisac długości kroku, dla których byla optymalizacja, 
 i jesli dobrze rozumiem to zliczyc(?) ile bylo minimow globalnych
 i dla nich obliczyc wartosci srednie dla x1*, x2*, y* i liczby wywolan f.celu
 jeszcze nie wiem jak, ale mozna zrobic jakas funkcje ktora oblicza te wartosci srednie dla minimow globalnych
 i wpisac recznie tw wyniki
'''

'''
Wykres
na razie myśle jak sie najlatwiej dostac do tego zeby latwo to wrzucic do excela
do tego, czyli nr iteracji i jakie wychodza wartosci x1* i x2* dla dwoch metod
jakis zaczatek tego jest wewnatrz metody rosenbrocka, ale cos sie wywala XD
'''

'''
Tabela 3
zwiazana jest juz z problemem rzeczywistym, mozna wpisac recznie,
 wszystko co potrzebne bedzie wyprintowane
 (wyniki do konsultacji)
'''
# Optymalizacja metodą Hooke-Jeevesa i Rosenbrocka, flagi na Flase, zeby zwrocilo tylko wartosc Q
wynik_hooke, liczba_wywolan_hooke = hooke_jeeves(
    funkcja_celu, punkt_startowy, krok_startowy, alfa_hooke, epsilon, maks_wywolan
)
Q_hooke = funkcja_celu(wynik_hooke, return_full=False)

wynik_rosen, liczba_wywolan_rosen = rosenbrock(
    funkcja_celu, punkt_startowy, kroki_startowe, alfa_rosen, beta, epsilon, maks_wywolan
)
Q_rosen = funkcja_celu(wynik_rosen, return_full=False)

# Wyświetlenie wyników
print("Wyniki optymalizacji dla problemu rzeczywistego:   ")
print("Metoda Hooke-Jeevesa: ")
print("Dla długości kroku:", krok_startowy)
print(f"k1* = {wynik_hooke[0]}, k2* = {wynik_hooke[1]}")
print(f"Q* = {Q_hooke}")
print(f"Liczba wywołań funkcji celu: {liczba_wywolan_hooke}")

print("Metoda Rosenbrocka: ")
print("Dla długości kroku:", kroki_startowe)
print(f"k1* = {wynik_rosen[0]}, k2* = {wynik_rosen[1]}")
print(f"Q* = {Q_rosen}")
print(f"Liczba wywołań funkcji celu: {liczba_wywolan_rosen}")
print("")
'''
Symulacje
Dla znalezionych parametrow obiema metodami przeprowadzamy symulację,
jak cos to w funkcji celu zwracam Q, albo czas, alpha i omega,
w zaleznosci od flagi -> dodane po to, zeby nie rozdzielac juz tej funkcji celu na dwie funkcje i zeby nie generowalo bledu
(przy zwracaniu returnem wszystkich wartosci, wywalalo sie w metodzie hooke_jeeves, mimo kombinowania,
 najlepszym rozwiazaniem jest flaga)
 wszystko to co wypisuje nizej ma byc wpisywane w arkuszu excela
 printuje to dla sprawdzenia wynikow i konsultacji
'''

czas_HJ, alpha_HJ, omega_HJ = funkcja_celu(wynik_hooke, return_full=True)
czas_rosen, alpha_rosen, omega_rosen = funkcja_celu(wynik_rosen, return_full=True)
print("Wyniki symulacji dla metody Hooke-Jeevesa:  ")
print("Czas (t), Kąt (α), Prędkość kątowa (ω)")
for t, a, w in zip(czas_HJ[:100], alpha_HJ[:100], omega_HJ[:100]):  # Ograniczenie do 100 próbek
    print(f"{t:.2f}, {a:.6f}, {w:.6f}")

print("")

print("Wyniki symulacji dla metody Rosenbrocka: ")
print("Czas (t), Kąt (α), Prędkość kątowa (ω)")
for t, a, w in zip(czas_rosen[:100], alpha_rosen[:100], omega_rosen[:100]):  # Ograniczenie do 100 próbek
    print(f"{t:.2f}, {a:.6f}, {w:.6f}")

"""
Rysowanie wykresu 3D dla funkcji testowej w zakresie x od -1 do 1
"""
# x = np.linspace(-1, 1, 201)
# x1_grid, x2_grid = np.meshgrid(x, x)
# z = x1_grid**2 + x2_grid**2 - np.cos(2.5 * np.pi * x1_grid) - np.cos(2.5 * np.pi * x2_grid) + 2
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x1_grid, x2_grid, z, cmap="viridis_r")
#
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('f(x1, x2)')
# ax.set_title('Wykres funkcji testowej')
#
# plt.show()
