import numpy as np
import pandas as pd
import os

input_dir = "./Data_npy/step2_vector_sequences"
output_dir = "./Data_npy/step3_histogram_features"
os.makedirs(output_dir, exist_ok=True)
output_base = os.path.join(output_dir, 'converted_csv')  # Base folder for output
os.makedirs(output_base, exist_ok=True)

# computes the histograms
def compute_histogram_features(vectors, bins=10):
    features = []

    # 1D Histograms
    theta = vectors[:, 3]
    delta_theta = np.diff(theta)
    r = vectors[:, 2]

    hist_theta, _ = np.histogram(theta, bins=bins, range=(-np.pi, np.pi), density=True)
    hist_dtheta, _ = np.histogram(delta_theta, bins=bins, range=(-np.pi, np.pi), density=True)
    hist_r, _ = np.histogram(r, bins=bins, range=(0, np.max(r)), density=True)

    features.extend(hist_theta)
    features.extend(hist_dtheta)
    features.extend(hist_r)

    # 2D Histogram: ⟨Φ1, R1⟩
    theta_cut = theta[:min(len(theta), len(r))]
    r_cut = r[:min(len(theta), len(r))]
    hist2d_phi1_r1, _, _ = np.histogram2d(
        theta_cut, r_cut, bins=bins, range=[[-np.pi, np.pi], [0, np.max(r)]]
    )
    features.extend(hist2d_phi1_r1.flatten())

    # 2D Histogram: ⟨Φ1, ΔΦ1⟩
    theta_cut2 = theta[:min(len(theta), len(delta_theta))]
    dtheta_cut2 = delta_theta[:min(len(theta), len(delta_theta))]
    hist2d_phi1_dphi1, _, _ = np.histogram2d(
        theta_cut2, dtheta_cut2, bins=bins, range=[[-np.pi, np.pi], [-np.pi, np.pi]]
    )
    features.extend(hist2d_phi1_dphi1.flatten())

    return np.array(features)


for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        vec = np.load(os.path.join(input_dir, filename))
        feature = compute_histogram_features(vec, bins=10)
        np.save(os.path.join(output_dir, filename), feature)

column_names = [
    "theta(2-11),delta_theta(12-21),r(22-31),2Dhistograms"
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