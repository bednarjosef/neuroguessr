# CALCULATE HOW MANY DATASET IMAGES ARE IN EACH H3 CELL

from datasets import load_dataset
from collections import Counter
from tqdm.auto import tqdm
import h3
import json

DATASET_NAME = 'josefbednar/streetview-acw-300k'
TRAIN_SPLIT = 'train'
H3_RESOLUTION = 2

def compute_h3_counts(dataset_name, split, res):
    ds = load_dataset(dataset_name, split=split)
    ds = ds.to_iterable_dataset(num_shards=393)
    ds = ds.select_columns(['latitude', 'longitude'])

    counts = Counter()

    for ex in tqdm(ds, desc=f'Counting H3 cells (res={res})'):
        lat = float(ex['latitude'])
        lon = float(ex['longitude'])
        h = h3.latlng_to_cell(lat, lon, res)
        counts[h] += 1

    return counts

if __name__ == '__main__':
    counts = compute_h3_counts(DATASET_NAME, TRAIN_SPLIT, H3_RESOLUTION)
    print(f'Unique non-empty H3 cells: {len(counts)}')

    # save counts to JSON
    with open('h3_utils/h3_counts_res2.json', 'w') as f:
        json.dump(counts, f)
