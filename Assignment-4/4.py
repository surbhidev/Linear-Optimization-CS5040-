# Annam Deddeepya Purnima (cs22btech11006)
# Mane Pooja Vinod (cs22btech11035)
# Surbhi (cs22btech11057)


import numpy as np
import pandas as pd
from scipy.linalg import null_space

def load_data(file_path):
    
    data = pd.read_csv(file_path, header=None)
    
    c = data.iloc[0, :-1].values
    
    A = data.iloc[1:, :-1].values
    
    b = data.iloc[1:, -1].values
    
    return c, A, b

def is_feasible_point(z, A, b):
    computed_b = np.dot(A, z)
    return np.all(computed_b <= b)

def adjust_to_basic_feasible(A, b, z, n):
    computed_b = np.dot(A, z)
    tight = np.isclose(computed_b, b)
    active = A[tight]

    while active.size == 0 or np.linalg.matrix_rank(active) < n:
        if active.size == 0:
            direction = np.random.rand(n)
        else:
            direction = null_space(active)[:, 0]
            if direction.size == 0:
                direction = np.random.rand(n)

        non_tight = ~tight
        A_non_tight = A[non_tight]
        b_non_tight = b[non_tight]
        computed_non_tight = computed_b[non_tight]

        denom = np.dot(A_non_tight, direction)
        valid = ~np.isclose(denom, 0)

        if np.any(valid): 
            step_sizes = (b_non_tight[valid] - computed_non_tight[valid]) / denom[valid]
            if step_sizes.size > 0:
                step = step_sizes[np.argmin(np.abs(step_sizes))]
                z += step * direction
            else:
                print("No valid step sizes found. Skipping adjustment.")
                break
        else:
            print("No valid directions found. Skipping adjustment.")
            break

        computed_b = np.dot(A, z)
        tight = np.isclose(computed_b, b)
        active = A[tight]

    return z


def handle_degeneracy(c, inverse_active, A, v, z, computed_v, tight):
    reduced_costs = np.dot(c, inverse_active)
    negative_indices = np.where((reduced_costs > 0) & ~np.isclose(reduced_costs, 0))[0]

    if negative_indices.size == 0:
        return None, None

    entering_index = np.argmin(negative_indices)
    direction = inverse_active[:, negative_indices[entering_index]]

    non_tight = ~tight
    A_non_tight = A[non_tight]
    v_non_tight = v[non_tight]
    computed_non_tight = computed_v[non_tight]

    denom = np.dot(A_non_tight, direction)
    
    valid = (denom > 0) & ~np.isclose(denom, 0)
    
    if not np.any(valid):
        print("No valid direction found. Terminating optimization.")
        return None, None
    
    return direction, valid


def simplex(c, A, b, z, n):
    m = A.shape[0]
    visited_vertices = [(z.copy(), np.dot(c, z))] 

    while True:
        computed_b = np.dot(A, z)
        tight = np.isclose(computed_b, b)
        active = A[tight]
        inverse_active = np.linalg.pinv(active)

        reduced_costs = np.dot(c, inverse_active)
        negative_indices = np.where((reduced_costs < 0) & ~np.isclose(reduced_costs, 0))[0]

        if negative_indices.size == 0:
            break

        direction, valid = handle_degeneracy(c, inverse_active, A, b, z, computed_b, tight)
        if direction is None:
            break

        non_tight = ~tight
        A_non_tight = A[non_tight]
        b_non_tight = b[non_tight]
        computed_non_tight = computed_b[non_tight]

        denom = np.dot(A_non_tight, direction)
        valid = (denom > 0) & ~np.isclose(denom, 0)

        if not np.any(valid):
            print("No valid directions found. Ending algorithm.")
            break

        step_sizes = (b_non_tight[valid] - computed_non_tight[valid]) / denom[valid]
        step = step_sizes[np.argmin(step_sizes)]
        z += step * direction
        
        visited_vertices.append((z.copy(), np.dot(c, z)))

    print("\nSequence of vertices visited:")
    for i, (vertex, cost) in enumerate(visited_vertices):
        print(f"Step {i + 1}: Vertex = {vertex}, Objective Value = {cost}")

    return z

if __name__ == "__main__":
    input_file_path = "test4.csv"  

    c, A, b = load_data(input_file_path)
    n = len(c)  
    m = len(b)  

    z = np.zeros(n)  

    if not is_feasible_point(z, A, b):
        print("The initial point is not feasible.")
        exit()

    print(f"Initial point: {z}, Initial Objective Value: {np.dot(c, z)}")

    feasible_z = adjust_to_basic_feasible(A, b, z, n)
    print(f"Feasible point: {feasible_z}, Feasible Objective Value: {np.dot(c, feasible_z)}")

    optimal_z = simplex(c, A, b, feasible_z, n)
    optimal_value = np.dot(c, optimal_z)

    print(f"\nFinal Optimal Point: {optimal_z}")
    print(f"Final Optimal Objective Value: {optimal_value}")