import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import torch

# --- CONFIGURATION ---
# We download the metadata from the original repo (it matches the WDS version)
REPO_ID = "osv5m/osv5m" 
FILENAME = "train.csv"

print(f"Downloading {FILENAME} from {REPO_ID}...")
try:
    # Download just the CSV (approx 500MB) - much faster than images!
    csv_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    print(f"Loaded metadata from: {csv_path}")
except Exception as e:
    print(f"Error downloading metadata: {e}")
    exit()

# Load into Pandas
print("Reading CSV (this may take a moment)...")
# We only need the 'country' column
df = pd.read_csv(csv_path, usecols=['country'])

# 1. Get Raw Counts
print("Counting classes...")
country_counts = df['country'].value_counts()
total_samples = len(df)
num_classes = len(country_counts)

print(f"Total Images: {total_samples:,}")
print(f"Total Countries found: {num_classes}")

# 2. Calculate Weights
# Method: Balanced Weighting
# Formula: Weight = Total_Samples / (Num_Classes * Class_Samples)
# This ensures that theoretically, every country contributes equally to the gradient.
class_weights = {}
raw_ratios = {}

# We normalize so the median weight is roughly 1.0 (keeps gradients stable)
# This prevents rare countries from having weights like 10,000 which explodes training
counts_array = country_counts.values
balanced_weights = total_samples / (num_classes * counts_array)
# Normalize to median = 1.0 for stability
normalized_weights = balanced_weights / np.median(balanced_weights)

for country, weight, count in zip(country_counts.index, normalized_weights, counts_array):
    class_weights[country] = weight
    raw_ratios[country] = count

# --- OUTPUT GENERATION ---

print("\n" + "="*60)
print(f"{'COUNTRY':<10} | {'IMAGES':<10} | {'CALC WEIGHT':<10}")
print("="*60)

# Print Top 10 (Most Frequent)
print("--- TOP 10 (Most Common) ---")
for country in country_counts.index[:10]:
    print(f"{country:<10} | {raw_ratios[country]:<10} | {class_weights[country]:.4f}")

print("\n--- BOTTOM 10 (Rarest) ---")
for country in country_counts.index[-10:]:
    print(f"{country:<10} | {raw_ratios[country]:<10} | {class_weights[country]:.4f}")

print("="*60)

# 3. Generate Python Code for your Training Script
print("\n\n--- COPY THIS INTO TRAIN.PY ---")
print("# Auto-generated Country List and Weights")
print("TARGET_COUNTRIES = [")
# Sort alphabetically for neatness in the list
sorted_countries = sorted(country_counts.index)
line_str = ""
for c in sorted_countries:
    line_str += f"'{c}', "
    if len(line_str) > 70:
        print(f"    {line_str}")
        line_str = ""
print(f"    {line_str}")
print("]")

print("\nweights = torch.tensor([")
line_str = ""
for c in sorted_countries:
    w = class_weights[c]
    line_str += f"{w:.4f}, "
    if len(line_str) > 70:
        print(f"    {line_str}")
        line_str = ""
print(f"    {line_str}")
print(f"]).to(DEVICE)")