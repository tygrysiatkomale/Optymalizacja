import numpy as np
import matplotlib.pyplot as plt


def test_fun(x1, x2):
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
    z = test_fun(x1_grid, x2_grid)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_grid, x2_grid, z, cmap="viridis_r")

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('Wykres funkcji testowej')

    plt.show()
    return 0


if __name__ == '__main__':
    make_test_function_chart()
    pass
