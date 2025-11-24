import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from model import GeoguessrModel  # <--- Import your new file

# --- CONFIG ---
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
STEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the countries you want to guess (The "Heads")
TARGET_COUNTRIES = ["FR", "US", "JP", "GB", "AU", "BR", "ZA", "KR", "ES", "IT"]
country_to_idx = {code: i for i, code in enumerate(TARGET_COUNTRIES)}

class WebDatasetStream(IterableDataset):
    def __init__(self, hf_dataset, transform, country_map):
        self.ds = hf_dataset
        self.transform = transform
        self.country_map = country_map

    def __iter__(self):
        for sample in self.ds:
            try:
                # Get metadata
                meta = sample["json"]
                country = meta.get('country')
                
                # Filter: Skip if country is unknown or not in our target list
                if country not in self.country_map:
                    continue
                
                # Process image
                img = sample["jpg"].convert("RGB")
                img_tensor = self.transform(img)
                label = self.country_map[country]
                
                yield img_tensor, label
            except Exception:
                continue


model = GeoguessrModel(num_classes=len(TARGET_COUNTRIES))
model = model.to(DEVICE)

data_config = model.get_config()
transforms = timm.data.create_transform(**data_config, is_training=True)

print("Connecting to WebDataset Stream...")
data_files = {"train": "hf://datasets/osv5m/osv5m-wds/train/*.tar"}
ds_stream = load_dataset("webdataset", data_files=data_files, split="train", streaming=True)

train_dataset = WebDatasetStream(ds_stream, transforms, country_to_idx)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

optimizer = optim.Adam(model.head.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print("Starting training...")
model.train()

for step, (imgs, labels) in enumerate(train_loader):
    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
    
    optimizer.zero_grad()
    outputs = model(imgs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if step % 1 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")
    
    if step >= STEPS:
        break

torch.save(model.state_dict(), "geoguessr.pth")
print("Saved model.")