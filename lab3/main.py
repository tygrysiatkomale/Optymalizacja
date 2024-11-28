import numpy as np
import matplotlib.pyplot as plt


def funkcja_testowa(x1, x2):
    """
    :epsilon: mega mala wartosc do unikniecia dzielenia przez zero
    :top: gorna wartosc dla funkcji testowej
    :bottom: dolna wartosc dla funkcji testowej
    """
    epsilon = 1e-15
    top = np.sin(np.pi * np.sqrt((x1 / np.pi)**2 + (x2 / np.pi)**2))
    bottom = (np.pi * np.sqrt((x1 / np.pi)**2 + (x2 / np.pi)**2)) + epsilon
    return top / bottom


def make_test_function_chart():
    x1, x2 = np.linspace(1, 6, 501), np.linspace(1, 6, 501)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    z = funkcja_testowa(x1_grid, x2_grid)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_grid, x2_grid, z, cmap="viridis_r")

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('Wykres funkcji testowej')

    plt.show()
    return 0


# testowa funkcjoa z zewnetrzna kara
def funkcja_testowa_external_penalty(x, a, c):
    """
    x: numpy array, decision variables [x1, x2]
    a: parameter for the norm constraint
    c: penalty coefficient
    """
    # Obliczenie testowej funkcji celu
    top = np.sin(np.pi * np.sqrt((x[0] / np.pi)**2 + (x[1] / np.pi)**2))
    bottom = np.pi * np.sqrt((x[0] / np.pi)**2 + (x[1] / np.pi)**2)
    result = top / bottom if bottom != 0 else 0  # Unikanie dzielenia przez zero

    # Inicjalizacja funkcji celu
    y = result

    # Dodanie zewnętrznej funkcji kary
    if -x[0] + 1 > 0:
        y += c * (-x[0] + 1)**2
    if -x[1] + 1 > 0:
        y += c * (-x[1] + 1)**2
    if np.linalg.norm(x) - a > 0:
        y += c * (np.linalg.norm(x) - a)**2

    return y

# f.testowa z wewnetrzna kara
def funkcja_testowa_internal_penalty(x, a, c):
    """
    x: numpy array, decision variables [x1, x2]
    a: parameter for the norm constraint
    c: penalty coefficient
    """
    # Obliczenie testowej funkcji celu
    top = np.sin(np.pi * np.sqrt((x[0] / np.pi)**2 + (x[1] / np.pi)**2))
    bottom = np.pi * np.sqrt((x[0] / np.pi)**2 + (x[1] / np.pi)**2)
    result = top / bottom if bottom != 0 else 0  # Unikanie dzielenia przez zero

    # Inicjalizacja funkcji celu
    y = result

    # Dodanie łagodniejszej wewnętrznej funkcji kary
    if -x[0] + 1 > 0 or -x[1] + 1 > 0 or np.linalg.norm(x) - a > 0:
        return 1e6  # Kara za wyjście poza obszar dopuszczalny (duża, ale nie ekstremalna)
    if -x[0] + 1 < 0:
        y -= c / (1 + (-x[0] + 1))  # Łagodniejsza kara za naruszenie ograniczenia
    if -x[1] + 1 < 0:
        y -= c / (1 + (-x[1] + 1))
    if np.linalg.norm(x) - a < 0:
        y -= c / (1 + (np.linalg.norm(x) - a))

    return y

# definiuje ograniczenia
def check_constraints(x, a):
    g1 = -x[0] + 1
    g2 = -x[1] + 1
    g3 = np.linalg.norm(x) - a
    return g1, g2, g3




if __name__ == '__main__':
    # make_test_function_chart()
    # pass
    x = np.array([0.5, 0.5])  # Wartości testowe zmiennych
    a = 5  # Parametr ograniczenia normy
    c = 10 # Współczynnik kary

    # Test zewnętrznej funkcji kary
    result_ext = funkcja_testowa_external_penalty(x, a, c)
    print("Zewnętrzna funkcja kary:", result_ext)

    # Test wewnętrznej funkcji kary
    result_int = funkcja_testowa_internal_penalty(x, a, c)
    print("Wewnętrzna funkcja kary:", result_int)

    constraints = check_constraints(x, a)
    print("Ograniczenia (g1, g2, g3):", constraints)
