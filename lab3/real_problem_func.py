from nelder_mead import nelder_mead  # Import własnej implementacji
import numpy as np

# dane do problemu rzeczywistego
# stale fizyczne  i parametry problemu
g = 9.81  # Przyspieszenie grawitacyjne, m/s^2
m = 0.6   # Masa piłki, kg
r = 0.12  # Promień piłki, m
rho = 1.2  # Gęstość powietrza, kg/m^3
C_d = 0.47  # Współczynnik oporu aerodynamicznego
S = np.pi * r**2  # Pole przekroju poprzecznego, m^2
t_max = 7  # Maksymalny czas symulacji, s
dt = 0.01  # Krok czasowy symulacji, s
time_steps = np.arange(0, t_max + dt, dt)  # Kroki czasowe symulacji

# poczatkowe warunki i ograniczenia
bounds = [(-10, 10), (-15, 15)]  # Ograniczenia dla v0 i omega
initial_v0, initial_omega = 5, 10  # Początkowe wartości prędkości i prędkości kątowej

def forces(vx, vy, omega):
    v = np.sqrt(vx**2 + vy**2)
    drag_x = -0.5 * C_d * rho * S * v * vx
    drag_y = -0.5 * C_d * rho * S * v * vy
    magnus_x = 0.5 * rho * omega * np.pi * r**2 * vy
    magnus_y = -0.5 * rho * omega * np.pi * r**2 * vx
    return drag_x, drag_y, magnus_x, magnus_y

def simulate_trajectory(v0, omega, t_steps):
    x, y = 0, 100
    vx, vy = v0, 0
    trajectory = []

    for t in t_steps:
        drag_x, drag_y, magnus_x, magnus_y = forces(vx, vy, omega)
        ax = (drag_x + magnus_x) / m
        ay = (drag_y + magnus_y - m * g) / m

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        trajectory.append((t, x, y))
        if y <= 0:
            break

    return np.array(trajectory)

def objective_real_world(params):
    v0, omega = params
    traj = simulate_trajectory(v0, omega, time_steps)
    y_50 = traj[np.abs(traj[:, 2] - 50).argmin(), 1]
    penalty = 0
    if not (4.5 <= y_50 <= 5.5):
        penalty += 1000 * min(abs(y_50 - 4.5), abs(y_50 - 5.5))
    return -traj[-1, 1] + penalty

def optimize_real_world(initial_v0, initial_omega, bounds, step_size=1.0, tol=1e-6, max_iter=1000):
    """
    Optymalizacja problemu rzeczywistego przy użyciu własnej implementacji Neldera-Meada.
    """
    def bounded_objective(x):
        # Nakładamy ograniczenia na x
        if not (bounds[0][0] <= x[0] <= bounds[0][1]) or not (bounds[1][0] <= x[1] <= bounds[1][1]):
            return 1e6  # Kara za wyjście poza zakres
        return objective_real_world(x)

    best_point, best_value, iterations = nelder_mead(
        bounded_objective,
        np.array([initial_v0, initial_omega]),
        step_size=step_size,
        tol=tol,
        max_iter=max_iter
    )
    return best_point, best_value, iterations
