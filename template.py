import numpy as np
import os
from collections import defaultdict
import pandas as pd

input_dir = "./Data_npy/step3_histogram_features"
output_dir = "./Data_npy/step4_templates"
os.makedirs(output_dir, exist_ok=True)
output_base = os.path.join(output_dir, 'converted_csv')  # Base folder for output
os.makedirs(output_base, exist_ok=True)

user_features = defaultdict(list)

for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        uid = filename.split("S")[0]  # e.g., U1
        sid = int(filename.split("S")[1].replace(".npy", ""))
        if 1 <= sid <= 20:  # Genuine training samples
            data = np.load(os.path.join(input_dir, filename))
            user_features[uid].append(data)

# Generate template
for uid, vectors in user_features.items():
    stacked = np.stack(vectors)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    quant_step = 1.5 * std

    np.save(os.path.join(output_dir, f"{uid}_template.npy"), mean)
    np.save(os.path.join(output_dir, f"{uid}_qstep.npy"), quant_step)


for filename in os.listdir(output_dir):
    if filename.endswith('.npy'):
        npy_path = os.path.join(output_dir, filename)
        array = np.load(npy_path)
        df = pd.DataFrame(array)

        # Save the CSV inside the output_base folder
        csv_path = os.path.join(output_base, filename.replace('.npy', '.csv'))
        df.to_csv(csv_path, index=False, header=False)

        # print(f"Converted {filename} → {csv_path}")