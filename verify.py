import numpy as np
import os

# === CONFIGURATION ===
user_id = "U1"             # e.g., "U1"
sample_id = 23            # Sample number (6–40)
threshold_file = "./Data_npy/step6_eer.txt"
feature_dir = "./Data_npy/step3_histogram_features"
template_dir = "./Data_npy/step4_templates"

# === Load EER Threshold ===
with open(threshold_file, "r") as f:
    line = f.readline()
    eer_threshold = float(line.strip().split("Threshold = ")[-1])

# === Construct file paths ===
sample_filename = f"{user_id}S{str(sample_id)}.npy"
feature_path = os.path.join(feature_dir, sample_filename)
template_path = os.path.join(template_dir, f"{user_id}_template.npy")
qstep_path = os.path.join(template_dir, f"{user_id}_qstep.npy")

# === Load data ===
fvec = np.load(feature_path)
template = np.load(template_path)
qstep = np.load(qstep_path)

    # === Quantize both template and input vector ===
quant_fvec = np.floor(fvec / (qstep + 1e-5))
quant_template = np.floor(template / (qstep + 1e-5))

    # === Compute Manhattan Distance ===
score = np.sum(np.abs(quant_fvec - quant_template))

    # === Verification decision ===
print(f"[INFO] Match score for {user_id}S{sample_id:02d}: {score:.2f}")
if score <= -eer_threshold:  # Because EER threshold was calculated on -scores
    print(f"[RESULT] ✅ Signature ACCEPTED as GENUINE")
else:
    print(f"[RESULT] ❌ Signature REJECTED as FORGERY")
