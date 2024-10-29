# plik z funkcja testowa, z rownaniem rozniczkowym, funkcja celu
import numpy as np
from scipy.integrate import solve_ivp

# funkcja testowa
def ff1T(x):
    return -np.cos(0.1 * x) * np.exp(-(0.1 * x - 2 * np.pi) ** 2) + 0.002 * (0.1 * x) ** 2

# funkcja celu
def ff2R(DA, T_target=50):
    """
   Funkcja celu do optymalizacji otworu DA, aby temperatura w zbiorniku B nie przekraczała 50 stopni C.

   Parametry:
   - DA: wielkość otworu w zbiorniku A
   - T_target: docelowa maksymalna temperatura w zbiorniku B

   Zwraca:
   - Różnicę między maksymalną temperaturą w B a wartością docelową (50°C).
   """
    # Parametry początkowe
    Y0 = [5, 1, 20]  # początkowe wartości VA, VB, TB

    # Rozwiązywanie równań różniczkowych w przedziale czasu [0, 2000]
    sol = solve_ivp(df1, [0, 2000], Y0, args=(DA,), dense_output=True)

    # Wyodrębnienie rozwiązania dla temperatury TB
    TB_values = sol.y[2]  # Trzecia wartość w wektorze Y to TB

    # Obliczenie maksymalnej temperatury w B
    max_TB = np.max(TB_values)

    # Funkcja celu: różnica między maksymalną temperaturą a wartością docelową
    return abs(max_TB - T_target)



# funkcja opisujaca rownanie rozniczkowe
def df1(t, Y, DA):
    """
    Funkcja zwracająca pochodne VA', VB' oraz TB' dla równań różniczkowych.

    Parametry:
    - t: czas (argument wymagany przez solver równań różniczkowych)
    - Y: wektor stanu w danym momencie (VA, VB, TB)
    - DA: wielkość otworu w zbiorniku A

    Zwraca:
    - dY: lista pochodnych [VA', VB', TB']
    """
    # Parametry stałe
    a = 0.98          # współczynnik lepkości
    b = 0.63          # współczynnik zwężenia strumienia
    g = 9.81          # przyspieszenie grawitacyjne
    PA = 0.5          # pole powierzchni zbiornika A
    PB = 1.0          # pole powierzchni zbiornika B
    DB = 0.00365665   # wielkość otworu w zbiorniku B
    Fin = 0.01        # ilość wody wpływającej do B
    Tin = 20.0        # temperatura wody wpływającej do B
    TA = 90.0         # temperatura wody w zbiorniku A

    # Wektor stanu w danym momencie
    VA = Y[0]         # objętość w zbiorniku A
    VB = Y[1]         # objętość w zbiorniku B
    TB = Y[2]         # temperatura w zbiorniku B

    # Obliczenie przepływów z A i B
    FAout = a * b * DA * np.sqrt(2 * g * VA / PA) if VA > 0 else 0
    FBout = a * b * DB * np.sqrt(2 * g * VB / PB) if VB > 0 else 0

    # Obliczenie pochodnych
    dVA_dt = -FAout                                  # VA' (zmiana objętości w A)
    dVB_dt = FAout + Fin - FBout                     # VB' (zmiana objętości w B)
    dTB_dt = (FAout / VB * (TA - TB) + Fin / VB * (Tin - TB)) if VB > 0 else 0  # TB' (zmiana temperatury w B)

    return [dVA_dt, dVB_dt, dTB_dt]