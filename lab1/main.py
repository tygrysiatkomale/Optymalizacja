import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
import random
import user_func
import algorithms


def make_plot(f, range_start=-100, range_stop=100):
    x = []
    y = []

    for i in range(range_start, range_stop):
        x.append(i)
        y.append(f(i))

    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Wartości')

    plt.title('Wykres funkcji')
    plt.xlabel('Oś X')
    plt.ylabel('Oś Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    return 0


def add_to_excel(x0, a, b, n, fib_ncalls, fib_min, line):
    file_name = "xlsx1.xlsx"
    try:
        # Próba załadowania istniejącego pliku
        workbook = load_workbook(file_name)
    except FileNotFoundError:
        # Jeśli plik nie istnieje, tworzymy nowy
        workbook = Workbook()

    # Wybieramy aktywny arkusz (pierwszy z dostępnych)
    sheet = workbook.active

    sheet[f'C{line}'] = x0
    sheet[f'D{line}'] = a
    sheet[f'E{line}'] = b
    sheet[f'F{line}'] = n
    sheet[f'G{line}'] = fib_min
    sheet[f'H{line}'] = user_func.ff1T(fib_min)
    sheet[f'I{line}'] = fib_ncalls
    sheet[f'J{line}'] = "lokalne" if user_func.ff1T(fib_min) > -0.1 else "globalne"

    # Zapisujemy zmiany w pliku
    workbook.save(file_name)
    print(f"Wartości zostały zapisane do pliku {file_name} w linii {line}")


# Parametry i testy
x0 = random.randint(-100,100)
d = 1
alpha = 1.5
nmax = 100
epsilon = 0.01

make_plot(user_func.ff1T)

for i in range(3, 103):
    x0 = random.randint(-100, 100)

    expansion_result = algorithms.expansion_method(user_func.ff1T, x0, d, alpha, nmax)
    print("Przedział ekspansji: ", expansion_result)
    a, b, fcalls = expansion_result

    fib_result = algorithms.fibonacci_method(user_func.ff1T, a, b, epsilon)
    print("Przybliżone minimum Fibonacciego: ", fib_result[0], "liczba wywolan:", fib_result[1])

    add_to_excel(x0, a, b, fcalls, fib_result[1], fib_result[0], i)

# Test metody Lagrange'a
c = (a + b) / 2
lagrange_result = algorithms.lagrange_interpolation(user_func.ff1T, a, b, c, epsilon)
print("Przybliżone minimum Lagrange'a: ", lagrange_result)
