import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Constants and parameters
g = 9.81  # Gravity, m/s^2
m = 0.6   # Mass of the ball, kg
r = 0.12  # Radius of the ball, m
rho = 1.2  # Air density, kg/m^3
C_d = 0.47  # Drag coefficient
S = np.pi * r**2  # Cross-sectional area, m^2
a_values = [4, 4.4934, 5]  # Values of 'a' for test function optimization

# Time parameters for simulation
t_max = 7
dt = 0.01
time_steps = np.arange(0, t_max + dt, dt)

# Test function parameters and constraints
def test_function(x, a):
    x1, x2 = x
    return np.sin(np.pi * x1) + np.pi * x2

def test_constraints(x, a):
    x1, x2 = x
    constraints = [
        -x1 + 1,  # g1(x)
        -x2 + 1,  # g2(x)
        x1 + x2 - a  # g3(x)
    ]
    return constraints

def penalty_function(x, a, penalty_type='outer', penalty_weight=1000):
    constraints = test_constraints(x, a)
    if penalty_type == 'outer':
        return penalty_weight * sum(max(0, g)**2 for g in constraints)
    elif penalty_type == 'inner':
        return penalty_weight * sum(-1 / max(0.001, g) for g in constraints)

def penalized_objective(x, a, penalty_type='outer', penalty_weight=1000):
    return test_function(x, a) + penalty_function(x, a, penalty_type, penalty_weight)

# Nelder-Mead optimization for the test function
def optimize_test_function(a, num_trials=100):
    results = []
    for _ in range(num_trials):
        # Generate a random initial feasible point
        x0 = np.random.uniform(0, 1, 2)
        x0[1] = min(x0[1], a - x0[0])  # Ensure x1 + x2 <= a
        res = minimize(penalized_objective, x0, args=(a, 'outer', 1000), method='Nelder-Mead', tol=1e-3)
        if res.success:
            x_opt = res.x
            f_val = test_function(x_opt, a)
            dist_from_origin = np.linalg.norm(x_opt)
            results.append((x_opt[0], x_opt[1], f_val, dist_from_origin))
    return results

# Perform optimizations for each value of 'a'
test_optimization_results = {a: optimize_test_function(a) for a in a_values}

# Calculate averages for Table 2
averages = {}
for a, results in test_optimization_results.items():
    df = pd.DataFrame(results, columns=['x1', 'x2', 'f(x)', 'distance'])
    averages[a] = df.mean().to_dict()

# Prepare test function results for Excel
test_table_1 = {a: pd.DataFrame(results, columns=['x1', 'x2', 'f(x)', 'distance']) for a, results in test_optimization_results.items()}
test_table_2 = pd.DataFrame(averages).T

def detailed_optimization(a, penalty_type='outer'):
    results = []
    for _ in range(100):  # Perform 100 optimizations
        # Generate a random initial feasible point
        x0 = np.random.uniform(0, 1, 2)
        x0[1] = min(x0[1], a - x0[0])  # Ensure x1 + x2 <= a

        # Optimization using Nelder-Mead
        func_calls = [0]  # Track function calls
        def tracked_penalized_objective(x):
            func_calls[0] += 1
            return penalized_objective(x, a, penalty_type, 1000)

        res = minimize(tracked_penalized_objective, x0, method='Nelder-Mead', tol=1e-3)
        
        if res.success:
            x_opt = res.x
            f_val = test_function(x_opt, a)
            dist_from_origin = np.linalg.norm(x_opt)  # r*
            y_val = f_val  # Here, y* is the same as f(x) for this problem
            results.append((x0[0], x0[1], x_opt[0], x_opt[1], dist_from_origin, y_val, func_calls[0]))
    return results

# Collect results for both outer and inner penalty methods
detailed_results = {}
for a in a_values:
    outer_results = detailed_optimization(a, penalty_type='outer')
    inner_results = detailed_optimization(a, penalty_type='inner')
    detailed_results[a] = {'outer': outer_results, 'inner': inner_results}

# Prepare data for Excel
detailed_table_1 = {}
for a, methods in detailed_results.items():
    outer_df = pd.DataFrame(methods['outer'], columns=[
        'x1(0)', 'x2(0)', 'x1*', 'x2*', 'r*', 'y*', 'Function Calls'
    ])
    inner_df = pd.DataFrame(methods['inner'], columns=[
        'x1(0)', 'x2(0)', 'x1*', 'x2*', 'r*', 'y*', 'Function Calls'
    ])
    detailed_table_1[a] = {'outer': outer_df, 'inner': inner_df}

# Save detailed results for Table 1 to Excel
detailed_table_path = "Tabela1.xlsx"
with pd.ExcelWriter(detailed_table_path) as writer:
    for a, method_results in detailed_table_1.items():
        method_results['outer'].to_excel(writer, sheet_name=f'Outer_a_{a}', index=False)
        method_results['inner'].to_excel(writer, sheet_name=f'Inner_a_{a}', index=False)

detailed_table_path

def forces(vx, vy, omega):
    v = np.sqrt(vx**2 + vy**2)
    drag_x = -0.5 * C_d * rho * S * v * vx
    drag_y = -0.5 * C_d * rho * S * v * vy
    magnus_x = 0.5 * rho * omega * np.pi * r**2 * vy
    magnus_y = -0.5 * rho * omega * np.pi * r**2 * vx
    return drag_x, drag_y, magnus_x, magnus_y

# Ball trajectory simulation
def simulate_trajectory(v0, omega, t_steps):
    # Initial conditions
    x, y = 0, 100
    vx, vy = v0, 0
    trajectory = []

    for t in t_steps:
        drag_x, drag_y, magnus_x, magnus_y = forces(vx, vy, omega)
        ax = (drag_x + magnus_x) / m
        ay = (drag_y + magnus_y - m * g) / m

        # Update velocities and positions
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        trajectory.append((t, x, y))
        if y <= 0:  # Stop if the ball hits the ground
            break

    return np.array(trajectory)

# Objective function for optimization: maximize x at y=50
def objective_real_world(params):
    v0, omega = params
    func_calls[0] += 1  # Track function calls
    traj = simulate_trajectory(v0, omega, time_steps)
    y_50 = traj[np.abs(traj[:, 2] - 50).argmin(), 1]  # Find x for closest y to 50
    penalty = 0
    if not (4.5 <= y_50 <= 5.5):  # Constraint: x at y=50 should be within [4.5, 5.5]
        penalty += 1000 * (min(abs(y_50 - 4.5), abs(y_50 - 5.5)))
    return -traj[-1, 1] + penalty  # Negative x_final for maximization

# Bounds for v0 and omega
bounds = [(-10, 10), (-15, 15)]

initial_v0, initial_omega = 5, 10

# Track function calls
func_calls = [0]

# Perform optimization for real-world problem
real_world_result = minimize(objective_real_world, [initial_v0, initial_omega], 
                              bounds=bounds, method='Nelder-Mead', tol=1e-3)

# Extract optimization results
optimized_v0, optimized_omega = real_world_result.x
final_x = -real_world_result.fun  # Since we minimized the negative x_final

# Simulate the trajectory with optimized values
trajectory = simulate_trajectory(optimized_v0, optimized_omega, time_steps)

# Save results for real-world problem to Excel
real_world_table_path = "Tabela3Symulacja.xlsx"
real_world_params = pd.DataFrame({
    'v0x(0)': [initial_v0],
    'ω(0)': [initial_omega],
    'vox*': [optimized_v0],
    'ω*': [optimized_omega],
    'xend*': [final_x],
    'Liczba wywołań funkcji celu': [func_calls[0]]
})

trajectory_df = pd.DataFrame(trajectory, columns=['Time (s)', 'X (m)', 'Y (m)'])

with pd.ExcelWriter(real_world_table_path) as writer:
    real_world_params.to_excel(writer, sheet_name='Table_3_Parameters', index=False)
    trajectory_df.to_excel(writer, sheet_name='Simulation_Trajectory', index=False)

real_world_table_path