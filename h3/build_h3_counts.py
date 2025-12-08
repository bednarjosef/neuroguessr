# build_h3_counts.py
from datasets import load_dataset
from collections import Counter
from tqdm.auto import tqdm
import h3
import json

DATASET_NAME = "josefbednar/world-streetview-500k"  # TODO
TRAIN_SPLIT = "train"                        # or whatever you called it
H3_RESOLUTION = 2                            # try 4 or 5

def compute_h3_counts(dataset_name, split, res):
    # Stream the dataset so it doesn't load everything into RAM
    ds = load_dataset(dataset_name, split=split)
    ds = ds.to_iterable_dataset(num_shards=393)
    # Only need these columns
    ds = ds.select_columns(["latitude", "longitude"])

    counts = Counter()

    for ex in tqdm(ds, desc=f"Counting H3 cells (res={res})"):
        lat = float(ex["latitude"])
        lon = float(ex["longitude"])
        h = h3.latlng_to_cell(lat, lon, res)
        counts[h] += 1

    return counts

if __name__ == "__main__":
    counts = compute_h3_counts(DATASET_NAME, TRAIN_SPLIT, H3_RESOLUTION)
    print(f"Unique non-empty H3 cells: {len(counts)}")

    # Save counts to JSON
    with open("h3_counts_res2.json", "w") as f:
        json.dump(counts, f)
