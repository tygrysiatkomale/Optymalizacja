import numpy as np

def nelder_mead(f, x_start, step_size=1.0, tol=1e-6, max_iter=1000, alpha=1.0, beta=0.5, gamma=2.0, delta=0.5):
    """
    Implementacja algorytmu Neldera-Meada.
    f: Funkcja celu, którą minimalizujemy
    x_start: Punkt początkowy (wektor numpy)
    step_size: Rozmiar początkowego sympleksu
    tol: Tolerancja zbieżności
    max_iter: Maksymalna liczba iteracji
    alpha: Współczynnik odbicia
    beta: Współczynnik kontrakcji
    gamma: Współczynnik ekspansji
    delta: Współczynnik redukcji
    """
    n = len(x_start)  # Liczba wymiarów
    simplex = [x_start + step_size * (np.eye(n)[i] if i < n else np.zeros(n)) for i in range(n + 1)]
    simplex = np.array(simplex)

    # Obliczamy wartości funkcji celu dla wierzchołków sympleksu
    f_values = np.array([f(x) for x in simplex])
    num_iter = 0

    while num_iter < max_iter:
        # Sortujemy wierzchołki według wartości funkcji celu
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]

        # Sprawdzamy kryterium zbieżności
        if np.max(f_values) - np.min(f_values) < tol:
            break

        # Obliczamy środek sympleksu (bez najgorszego wierzchołka)
        centroid = np.mean(simplex[:-1], axis=0)

        # Odbicie
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)

        if f_values[0] <= fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
            continue

        # Ekspansja
        if fr < f_values[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
            continue

        # Kontrakcja
        xc = centroid + beta * (simplex[-1] - centroid)
        fc = f(xc)
        if fc < f_values[-1]:
            simplex[-1] = xc
            f_values[-1] = fc
            continue

        # Redukcja
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + delta * (simplex[i] - simplex[0])
            f_values[i] = f(simplex[i])

        num_iter += 1

    # Zwracamy najlepszy punkt i jego wartość funkcji celu
    best_idx = np.argmin(f_values)
    return simplex[best_idx], f_values[best_idx], num_iter
