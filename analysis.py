import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load match scores
df = pd.read_csv("./Data_npy/step5_scores/match_scores.csv")

# Separate genuine and forgery scores
genuine_scores = df[df["label"] == "genuine"]["score"].values
forgery_scores = df[df["label"] == "forgery"]["score"].values

# Save score arrays
np.save("./Data_npy/step6_genuine_scores.npy", genuine_scores)
np.save("./Data_npy/step6_forgery_scores.npy", forgery_scores)

# 1. Plot Genuine Scores Histogram
plt.figure(figsize=(8, 4))
plt.hist(genuine_scores, bins=30, alpha=0.7, color='skyblue', label='Genuine')
plt.title("Genuine Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.savefig("./Data_npy/step6_genuine_histogram.png")
plt.close()

# 2. Plot Forgery Scores Histogram
plt.figure(figsize=(8, 4))
plt.hist(forgery_scores, bins=30, alpha=0.7, color='orange', label='Forgery')
plt.title("Forgery Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.savefig("./Data_npy/step6_forgery_histogram.png")
plt.close()


# Build labels and scores for ROC
labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(forgery_scores)])
scores = np.concatenate([genuine_scores, forgery_scores])

# Invert scores for ROC (lower = better)
fpr, tpr, thresholds = roc_curve(labels, -scores)
eer_threshold = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]

# Save EER
with open("./Data_npy/step6_eer.txt", "w") as f:
    f.write(f"EER = {eer:.4f}, Threshold = {eer_threshold:.2f}\n")

# ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.scatter([eer], [1 - eer], c='red', label=f"EER = {eer:.2f}")
plt.xlabel("False Acceptance Rate (FAR)")
plt.ylabel("True Acceptance Rate (1 - FRR)")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("./Data_npy/step6_roc_curve.png")
plt.close()


with open("./Data_npy/step6_stats.txt", "w") as f:
    f.write(f"Genuine Min: {genuine_scores.min()}, Max: {genuine_scores.max()}, Mean: {genuine_scores.mean()}\n")
    f.write(f"Forgery Min: {forgery_scores.min()}, Max: {forgery_scores.max()}, Mean: {forgery_scores.mean()}\n")
    f.write(f"EER = {eer:.4f}, Threshold = {eer_threshold:.2f}\n")
