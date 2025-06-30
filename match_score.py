import numpy as np
import os

feature_dir = "./Data_npy/step3_histogram_features"
template_dir = "./Data_npy/step4_templates"
output_dir = "./Data_npy/step5_scores"
os.makedirs(output_dir, exist_ok=True)

results = []

# finds the match scores for signature
for filename in os.listdir(feature_dir):
    if filename.endswith(".npy"):
        uid = filename.split("S")[0]
        sid = int(filename.split("S")[1].replace(".npy", ""))
        fvec = np.load(os.path.join(feature_dir, filename))

        template_path = os.path.join(template_dir, f"{uid}_template.npy")
        qstep_path = os.path.join(template_dir, f"{uid}_qstep.npy")

        if os.path.exists(template_path) and os.path.exists(qstep_path):
            template = np.load(template_path)
            qstep = np.load(qstep_path)
            quant_fvec = np.floor(fvec / (qstep + 1e-5))
            quant_template = np.floor(template / (qstep + 1e-5))

            score = np.sum(np.abs(quant_fvec - quant_template))
            label = "genuine" if sid <= 20 else "forgery"
            results.append(f"{filename},{uid},{sid},{label},{score}")

# Save scores
with open(os.path.join(output_dir, "match_scores.csv"), "w") as f:
    f.write("filename,user_id,sample_id,label,score\n")
    f.write("\n".join(results))

