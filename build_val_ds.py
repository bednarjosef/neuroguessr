import os
import glob
from collections import defaultdict

import webdataset as wds
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Value

# ---------------------------------------------------------
# 1. Config
# ---------------------------------------------------------

VAL_DIR = "./osv5m_local/val"   # directory containing val/*.tar
OUTPUT_DIR = "./osv5m_local/small_val"  # where to save the HF dataset
TARGET_PER_COUNTRY = 100

# Your list of countries (use the same one you used for clustering)
countries = ['AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG',
             'KH', 'CA', 'CL', 'CO', 'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI',
             'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS', 'IN', 'ID', 'IE', 'IL',
             'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY',
             'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA',
             'PE', 'PH', 'PL', 'PT', 'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS',
             'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE', 'CH', 'TW', 'TH', 'TR',
             'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN']

countries_set = set(countries)

# ---------------------------------------------------------
# 2. Build WebDataset pipeline over val/*.tar
# ---------------------------------------------------------

tar_files = sorted(glob.glob(os.path.join(VAL_DIR, "*.tar")))
if not tar_files:
    raise RuntimeError(f"No .tar files found in {VAL_DIR}")

print(f"Found {len(tar_files)} val shards")

dataset = (
    wds.WebDataset(tar_files, shardshuffle=True)
    .shuffle(10000)
    .decode("pil")
    .to_tuple("jpg", "json")  # -> (PIL.Image, dict)
)

# ---------------------------------------------------------
# 3. Collect up to 100 images per country
# ---------------------------------------------------------

buffers = defaultdict(list)  # country -> list of records
num_full = 0                 # how many countries already reached 100
already_full = set()

for img, meta in dataset:
    country = meta.get("country")
    if country not in countries_set:
        continue

    # skip if we already have enough for this country
    if country in already_full:
        continue

    lat = meta.get("latitude")
    lon = meta.get("longitude")
    if lat is None or lon is None:
        continue

    # store basic info
    buffers[country].append({
        "image": img.copy(),
        "country": country,
        "latitude": float(lat),
        "longitude": float(lon),
    })

    # if we just reached TARGET_PER_COUNTRY for this country, mark full
    if len(buffers[country]) == TARGET_PER_COUNTRY:
        already_full.add(country)
        num_full += 1
        print(f"Country {country}: reached {TARGET_PER_COUNTRY} samples "
              f"({num_full}/{len(countries_set)} full)")

        # If all countries are full, we can stop early
        if num_full == len(countries_set):
            print("Collected enough samples for all countries, stopping.")
            break

print("\nCollection finished.")
for c in sorted(countries_set):
    print(f"{c}: {len(buffers[c])} samples")

# ---------------------------------------------------------
# 4. Flatten into a single list of records
# ---------------------------------------------------------

records = []
for c in sorted(countries_set):
    records.extend(buffers[c])

print(f"Total samples in small val set: {len(records)}")

# ---------------------------------------------------------
# 5. Build Hugging Face Dataset and save to disk
# ---------------------------------------------------------

images = [r["image"] for r in records]
countries_col = [r["country"] for r in records]
lats = [r["latitude"] for r in records]
lons = [r["longitude"] for r in records]

features = Features({
    "image": HFImage(),
    "country": Value("string"),
    "latitude": Value("float32"),
    "longitude": Value("float32"),
})

hf_ds = Dataset.from_dict(
    {
        "image": images,
        "country": countries_col,
        "latitude": lats,
        "longitude": lons,
    },
    features=features,
)

print(hf_ds)
hf_ds.save_to_disk(OUTPUT_DIR)
print(f"\nSaved small val dataset to: {OUTPUT_DIR}")
