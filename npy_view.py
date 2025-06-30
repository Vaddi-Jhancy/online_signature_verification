import os
import numpy as np
import pandas as pd

folder_path = "./Data_npy"

# to view genuine scores and forgery scores in the csv file.
for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        npy_path = os.path.join(folder_path, filename)
        array = np.load(npy_path)
        df = pd.DataFrame(array)
        csv_path = npy_path.replace('.npy', '.csv')
        df.to_csv(csv_path, index=False, header=False)
        print(f"Converted {filename} to CSV.")
