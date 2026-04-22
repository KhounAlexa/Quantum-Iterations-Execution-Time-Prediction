#!/usr/bin/env python3
"""
Rebuild training dataset from existing aer.csv.
Regenerates:
  - data/training_data_log.npy      (log-transformed labels, 174D node features)
  - data/standardization_stats.pth  (normalization stats + non_zero_indices)
"""
import sys, os, pickle
import numpy as np
import torch
from pathlib import Path

sys.path.append(os.getcwd() + '/data_preparation')
from data_preparation import execution, helper

DATA_PATH = Path("datav1")

print("Loading final_combination.csv...", flush=True)
import pandas as pd
CSV_FILE = sys.argv[1] if len(sys.argv) > 1 else "combined_highq.csv"
csv_path = Path(CSV_FILE) if Path(CSV_FILE).is_absolute() or '/' in CSV_FILE else DATA_PATH / CSV_FILE
df = pd.read_csv(csv_path)
# Normalise to the columns generate_training_sample_execution_time expects
df = df[['quantum_circuit', 'time_taken']].copy()
df['device'] = 'aer'
print(f"  {len(df)} rows", flush=True)

print("\nExtracting features (this may take a while)...", flush=True)
sample, time_taken = execution.Execution().generate_training_sample_execution_time(df)
print(f"  {len(sample)} samples extracted", flush=True)

print("\nRefining global features (drop always-zero columns)...", flush=True)
refined, non_zero_indices = helper.refine_training_data(sample)
print(f"  Global features: {len(non_zero_indices)} kept", flush=True)

print("\nStandardizing features...", flush=True)
std_data, stats = helper.standardization_training_data(refined)

print(f"\nNode feature shape : {std_data[0].x.shape}")
print(f"Global feature shape: {std_data[0].global_features.shape}")

# Apply log transform to labels
print("\nApplying log transform to labels...", flush=True)
for i in range(len(std_data)):
    t = float(time_taken[i])
    std_data[i].y = float(np.log(t + 1e-8))

# Save dataset
out_dir = csv_path.parent
out_npy = out_dir / "training_data_log.npy"
with open(out_npy, "wb") as f:
    pickle.dump(std_data, f)
print(f"\nSaved: {out_npy}  ({len(std_data)} samples)", flush=True)

# Save normalization stats
out_pth = out_dir / "standardization_stats.pth"
torch.save({"stats": stats, "non_zero_indices": non_zero_indices}, str(out_pth))
print(f"Saved: {out_pth}", flush=True)


print("\nDone! Dataset ready.", flush=True)
