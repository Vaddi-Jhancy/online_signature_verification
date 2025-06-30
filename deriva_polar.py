import numpy as np
import os
import pandas as pd

input_dir = "./Data_npy/step1_parsed_signatures"
output_dir = "./Data_npy/step2_vector_sequences"
os.makedirs(output_dir, exist_ok=True)
output_base = os.path.join(output_dir, 'converted_csv')  # Base folder for output
os.makedirs(output_base, exist_ok=True)

# finds the derivatives
def compute_derivatives(data, order=2):
    derivs = [data]
    for _ in range(order):
        derivs.append(np.diff(derivs[-1], axis=0))
    return derivs[1:]

# constructs the vector sequences
def construct_vector_sequence(data, order=2):
    x, y, p = data[:, 0], data[:, 1], data[:, 6]

    x_derivs = compute_derivatives(x, order)
    y_derivs = compute_derivatives(y, order)
    p_derivs = compute_derivatives(p, order)

    # Use the smallest length across all derivatives
    min_len = min(len(d) for d in x_derivs + y_derivs + p_derivs)

    vectors = []
    for k in range(order):
        xk = x_derivs[k][:min_len]
        yk = y_derivs[k][:min_len]
        pk = p_derivs[k][:min_len]
        rk = np.sqrt(xk**2 + yk**2)
        thetak = np.arctan2(yk, xk)
        v_k = np.stack([xk, yk, rk, thetak, pk], axis=1)
        vectors.append(v_k)

    return np.concatenate(vectors, axis=1)


for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        data = np.load(os.path.join(input_dir, filename))
        vec_seq = construct_vector_sequence(data)
        np.save(os.path.join(output_dir, filename), vec_seq)

column_names = [
    "x1", "y1","r1", "theta1",
    "p1", "x2", "y2", "r2", "theta2", "p2"
]

for filename in os.listdir(output_dir):
    if filename.endswith('.npy'):
        npy_path = os.path.join(output_dir, filename)
        array = np.load(npy_path)
        df = pd.DataFrame(array , columns=column_names)

        # Save the CSV inside the output_base folder
        csv_path = os.path.join(output_base, filename.replace('.npy', '.csv'))
        df.to_csv(csv_path, index=False, header=True)

        # print(f"Converted {filename} → {csv_path}")