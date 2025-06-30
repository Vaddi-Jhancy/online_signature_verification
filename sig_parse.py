import os
import numpy as np
import pandas as pd

input_dir = "./Task2/Task2"
output_dir = "./Data_npy/step1_parsed_signatures"
os.makedirs(output_dir, exist_ok=True)
output_base = os.path.join(output_dir, 'converted_csv')  # Base folder for output
os.makedirs(output_base, exist_ok=True)

# parse the files
def parse_signature_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    data = [list(map(float, line.strip().split())) for line in lines[1:n+1]]
    return np.array(data)

for filename in os.listdir(input_dir):
    if filename.endswith(".TXT"):
        user_data = parse_signature_file(os.path.join(input_dir, filename))
        np.save(os.path.join(output_dir, filename.replace(".TXT", ".npy")), user_data)

column_names = [
    "X-coordinate", "Y-coordinate", "Timestamp",
    "Button Status", "Azimuth", "Altitude", "Pressure"
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
