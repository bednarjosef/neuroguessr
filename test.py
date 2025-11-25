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
    img = sample["jpg"]
    meta = sample["json"]

    country = meta['country']
    latitude = meta['latitude']
    longitude = meta['longitude']
    
    print(meta)

    # print(f"Sample {i}: Country={country} | Latitude={latitude} | Longitude={longitude}")
    if i >= 4:
        break
