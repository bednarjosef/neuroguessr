import os
import glob
from collections import defaultdict

import webdataset as wds
from PIL import Image
from datasets import Dataset, Features, Value

# ---------------- CONFIG ----------------
VAL_DIR = "./osv5m_local/val"               # val/*.tar from osv5m-wds
OUT_IMG_ROOT = "./osv5m_val_small_imgs"     # where to save small JPGs
OUT_HF_DIR = "./osv5m_val_small_hf"         # where to save HF dataset
TARGET_PER_COUNTRY = 100

countries = ['AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG',
             'KH', 'CA', 'CL', 'CO', 'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI',
             'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS', 'IN', 'ID', 'IE', 'IL',
             'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY',
             'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA',
             'PE', 'PH', 'PL', 'PT', 'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS',
             'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE', 'CH', 'TW', 'TH', 'TR',
             'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN']
countries_set = set(countries)

os.makedirs(OUT_IMG_ROOT, exist_ok=True)

# ------------- WDS over val/*.tar -------------
tar_files = sorted(glob.glob(os.path.join(VAL_DIR, "*.tar")))
if not tar_files:
    raise RuntimeError(f"No .tar files found in {VAL_DIR}")

print(f"Found {len(tar_files)} val shards")

dataset = (
    wds.WebDataset(tar_files, shardshuffle=True)
    .shuffle(10000)
    .decode("pil")
    .to_tuple("jpg", "json")   # -> (PIL.Image, dict)
)

# ------------- Collect + save as JPGs -------------
buffers = defaultdict(int)   # country -> count
records = []                 # rows for HF dataset

for img, meta in dataset:
    country = meta.get("country")
    if country not in countries_set:
        continue

    if buffers[country] >= TARGET_PER_COUNTRY:
        continue

    lat = meta.get("latitude")
    lon = meta.get("longitude")
    if lat is None or lon is None:
        continue

    # Save image to disk: ./osv5m_val_small_imgs/COUNTRY/idx.jpg
    country_dir = os.path.join(OUT_IMG_ROOT, country)
    os.makedirs(country_dir, exist_ok=True)

    idx = buffers[country]
    img_path = os.path.join(country_dir, f"{idx:04d}.jpg")
    img.convert("RGB").save(img_path, format="JPEG", quality=95)

    buffers[country] += 1

    records.append({
        "image_path": img_path,
        "country": country,
        "latitude": float(lat),
        "longitude": float(lon),
    })

    # optional: early exit if all countries reached target
    # if all(buffers[c] >= TARGET_PER_COUNTRY for c in countries_set):
    #     break

print("\nPer-country counts:")
total = 0
for c in sorted(countries_set):
    cnt = buffers[c]
    total += cnt
    print(f"{c}: {cnt} samples")

print(f"Total samples in small val set: {total}")

# ------------- HF dataset: just paths + metadata -------------
features = Features({
    "image_path": Value("string"),
    "country": Value("string"),
    "latitude": Value("float32"),
    "longitude": Value("float32"),
})

hf_ds = Dataset.from_list(records).cast(features)
print(hf_ds)

hf_ds.save_to_disk(OUT_HF_DIR)
print(f"\nSaved small val HF dataset to: {OUT_HF_DIR}")
