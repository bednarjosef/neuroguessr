from datasets import load_dataset

data_files = {
    "train": "hf://datasets/osv5m/osv5m-wds/train/*.tar",
}

ds = load_dataset(
    "webdataset",
    data_files=data_files,
    split="train",
    streaming=True,
)

# Peek a few samples
for i, sample in enumerate(ds):
    # typical keys in WebDataset: "jpg" for image, "json" for metadata
    img = sample["jpg"]      # a PIL.Image or array (depending on config)
    meta = sample["json"]    # dict with country, city, lat, lon, etc.

    country = meta['country']
    latitude = meta['latitude']
    longitude = meta['longitude']

    print(f"Sample {i}: Country={country} | Latitude={latitude} | Longitude={longitude}")
    if i >= 4:
        break
