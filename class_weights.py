import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import torch

# --- CONFIGURATION ---
# Only these countries will be counted and weighted
TARGET_COUNTRIES = ['AU', 'BR', 'CA', 'CZ', 'FR', 'ID', 'IN', 'JP', 'MX', 'RU']

# We download the metadata from the original repo
REPO_ID = "osv5m/osv5m" 
FILENAME = "train.csv"

print(f"Downloading {FILENAME} from {REPO_ID}...")
try:
    csv_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    print(f"Loaded metadata from: {csv_path}")
except Exception as e:
    print(f"Error downloading metadata: {e}")
    exit()

# Load into Pandas
print("Reading CSV...")
df = pd.read_csv(csv_path, usecols=['country'])

# 1. Filter for ONLY Target Countries
print(f"Filtering dataset for {len(TARGET_COUNTRIES)} target countries...")
# Keep only rows where country is in our list
df_subset = df[df['country'].isin(TARGET_COUNTRIES)]

# 2. Get Raw Counts
print("Counting classes...")
country_counts = df_subset['country'].value_counts()
total_subset_samples = len(df_subset)
num_classes = len(TARGET_COUNTRIES)

print(f"Total Images (Subset): {total_subset_samples:,}")

# Ensure we have counts for all targets (even if 0)
for country in TARGET_COUNTRIES:
    if country not in country_counts:
        country_counts[country] = 0

# 3. Calculate Weights
# Sort alphabetically to ensure the weights match your list order
sorted_countries = sorted(TARGET_COUNTRIES)

class_weights = {}
raw_counts = {}

# Get counts in sorted order
counts_array = np.array([country_counts[c] for c in sorted_countries])

# Avoid division by zero if a country has 0 images
safe_counts = counts_array.copy()
safe_counts[safe_counts == 0] = 1 

# Formula: Weight = Total / (Num_Classes * Count)
balanced_weights = total_subset_samples / (num_classes * safe_counts)

# Normalize to median = 1.0
normalized_weights = balanced_weights / np.median(balanced_weights)

for i, country in enumerate(sorted_countries):
    class_weights[country] = normalized_weights[i]
    raw_counts[country] = counts_array[i]

# --- OUTPUT GENERATION ---

print("\n" + "="*60)
print(f"{'COUNTRY':<10} | {'IMAGES':<10} | {'CALC WEIGHT':<10}")
print("="*60)

for country in sorted_countries:
    count = raw_counts[country]
    weight = class_weights[country]
    print(f"{country:<10} | {count:<10} | {weight:.4f}")

print("="*60)

# 4. Generate Python Code for your Training Script
print("\n\n--- COPY THIS INTO TRAIN.PY ---")
print("# Auto-generated Target List and Weights")
print("TARGET_COUNTRIES = [")
line_str = ""
for c in sorted_countries:
    line_str += f"'{c}', "
print(f"    {line_str}")
print("]")

print("\nweights = torch.tensor([")
line_str = ""
for c in sorted_countries:
    w = class_weights[c]
    line_str += f"{w:.4f}, "
print(f"    {line_str}")
print(f"]).to(DEVICE)")