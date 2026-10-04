import os
import json
import numpy as np

results_path = "outputs/evaluation/v2_validation_results.json"
with open(results_path, "r", encoding="utf-8") as f:
    records = json.load(f)

# Collect metrics
v1_wers, v2_wers, diff_wers = [], [], []
v1_mcds, v2_mcds, diff_mcds = [], [], []
v1_d1, v2_d1 = [], []
v1_d2, v2_d2 = [], []
v1_gv, v2_gv = [], []
v1_bd, v2_bd = [], []

for r in records:
    v1_wers.append(r["v1_wer"])
    v2_wers.append(r["v2_wer"])
    diff_wers.append(r["diff_wer"])
    
    v1_mcds.append(r["v1_mcd"])
    v2_mcds.append(r["v2_mcd"])
    diff_mcds.append(r["diff_mcd"])
    
    v1_d1.append(r["v1_diag"]["d1_mag"])
    v2_d1.append(r["v2_diag"]["d1_mag"])
    
    v1_d2.append(r["v1_diag"]["d2_mag"])
    v2_d2.append(r["v2_diag"]["d2_mag"])
    
    v1_gv.append(r["v1_diag"]["gv"])
    v2_gv.append(r["v2_diag"]["gv"])
    
    v1_bd.append(r["v1_diag"]["bound_disc"])
    v2_bd.append(r["v2_diag"]["bound_disc"])

def bootstrap_ci(data, num_samples=1000, alpha=0.05):
    data = np.array(data)
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(num_samples)]
    return np.percentile(means, [alpha/2 * 100, (1 - alpha/2) * 100])

print("=== V2 Validation Analysis (655 Utterances) ===")

print("\n--- WER (%) ---")
print(f"V1 Mean: {np.mean(v1_wers)*100:.2f}% | V2 Mean: {np.mean(v2_wers)*100:.2f}%")
print(f"Diff Mean: {np.mean(diff_wers)*100:.2f}% | Median: {np.median(diff_wers)*100:.2f}% | Std: {np.std(diff_wers)*100:.2f}%")
ci = bootstrap_ci(diff_wers)
print(f"Diff Bootstrap 95% CI: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")

print("\n--- MCD (dB) ---")
print(f"V1 Mean: {np.mean(v1_mcds):.2f} | V2 Mean: {np.mean(v2_mcds):.2f}")
print(f"Diff Mean: {np.mean(diff_mcds):.2f} | Median: {np.median(diff_mcds):.2f} | Std: {np.std(diff_mcds):.2f}")
ci = bootstrap_ci(diff_mcds)
print(f"Diff Bootstrap 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

print("\n--- Trajectory Diagnostics (Means) ---")
print(f"{'Metric':<30} | {'V1':<10} | {'V2':<10}")
print("-" * 55)
print(f"{'First-order change (d1)':<30} | {np.mean(v1_d1):<10.4f} | {np.mean(v2_d1):<10.4f}")
print(f"{'Second-order change (d2)':<30} | {np.mean(v1_d2):<10.4f} | {np.mean(v2_d2):<10.4f}")
print(f"{'Global Variance (GV)':<30} | {np.mean(v1_gv):<10.4f} | {np.mean(v2_gv):<10.4f}")
print(f"{'Boundary discontinuity':<30} | {np.mean(v1_bd):<10.4f} | {np.mean(v2_bd):<10.4f}")

# Check solver success
solver_successes = sum(1 for r in records if r["v2_solver_metrics"]["solver_success"])
print(f"\nMLPG Solver Success Rate: {solver_successes}/{len(records)} ({solver_successes/len(records)*100:.1f}%)")
