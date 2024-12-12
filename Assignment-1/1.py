# Annam Deddeepya Purnima (cs22btech11006)
# Mane Pooja Vinod (cs22btech11035)
# Surbhi (cs22btech11057)

import numpy as np
from scipy.linalg import null_space

def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    z = np.array([float(x) for x in lines[0].split()])

    c = np.array([float(x) for x in lines[1].split()])

    A = []
    v = []
    for line in lines[2:]:
        row = [float(x) for x in line.split()]
        A.append(row[:-1])  
        v.append(row[-1])   

    A = np.array(A)
    v = np.array(v)

    n = A.shape[1]  

    return n, z, c, A, v

def is_feasible_point(z, A, v):
    computed_v = np.dot(A, z)
    return np.all((computed_v < v) | np.isclose(computed_v, v))

def adjust_to_basic_feasible(A, v, z, n):
    computed_v = np.dot(A, z)
    tight = np.isclose(computed_v, v)
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
        v_non_tight = v[non_tight]
        computed_non_tight = computed_v[non_tight]

        denom = np.dot(A_non_tight, direction)
        valid = ~np.isclose(denom, 0)
        step_sizes = (v_non_tight[valid] - computed_non_tight[valid]) / denom[valid]

        step = step_sizes[np.argmin(np.abs(step_sizes))]
        z += step * direction
        
        computed_v = np.dot(A, z)
        tight = np.isclose(computed_v, v)
        active = A[tight]

    return z

def simplex(A, v, z, c, n):
    computed_v = np.dot(A, z)
    tight = np.isclose(computed_v, v)
    active = A[tight]
    inverse_active = np.linalg.pinv(active)
    visited_vertices = [(z.copy(), np.dot(c, z))] 

    while True:
        reduced_costs = np.dot(c, inverse_active)
        negative_indices = np.where((reduced_costs < 0) & ~np.isclose(reduced_costs, 0))[0]

        if negative_indices.size == 0:
            break
        
        direction = -inverse_active[:, negative_indices[0]]

        non_tight = ~tight
        A_non_tight = A[non_tight]
        v_non_tight = v[non_tight]
        computed_non_tight = computed_v[non_tight]

        denom = np.dot(A_non_tight, direction)
        valid = (denom > 0) & ~np.isclose(denom, 0)
        
        if not np.any(valid):
            print("No valid directions found. Ending algorithm.")
            break

        step_sizes = (v_non_tight[valid] - computed_non_tight[valid]) / denom[valid]

        if step_sizes.size == 0:
            print("No valid step sizes. Algorithm terminates.")
            break

        step = step_sizes[np.argmin(step_sizes)]
        z += step * direction
        computed_v = np.dot(A, z)
        tight = np.isclose(computed_v, v)
        active = A[tight]
        inverse_active = np.linalg.pinv(active)

        visited_vertices.append((z.copy(), np.dot(c, z)))

    print("\nSequence of vertices visited:")
    for i, (vertex, cost) in enumerate(visited_vertices):
        print(f"Step {i + 1}: Vertex = {vertex}, Cost = {cost}")

    return z

if __name__ == "__main__":
    input_file_path = "t.csv"
    n, z, c, A, v = load_data(input_file_path)

    if not is_feasible_point(z, A, v):
        print("The given point is not feasible.")
        exit()

    print(f"Initial point: {z}, Initial cost: {np.dot(c, z)}")

    feasible_z = adjust_to_basic_feasible(A, v, z, n)
    print(f"Feasible point: {feasible_z}, Feasible cost: {np.dot(c, feasible_z)}")
    
    optimal_z = simplex(A, v, feasible_z, c, n)
    optimal_cost = np.dot(c, optimal_z)

    print(f"\nFinal Optimal Point: {optimal_z}")
    print(f"Final Optimal Cost: {optimal_cost}")
