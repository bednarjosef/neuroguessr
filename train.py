import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds
from model import GeoguessrModel

# --- CONFIG ---
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
STEPS = 1000  # Number of steps before stopping (increase this for real training!)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Target Countries (Add more ISO codes here as needed)
TARGET_COUNTRIES = ["FR", "US", "JP", "GB", "AU", "BR", "ZA", "KR", "ES", "IT"]
country_to_idx = {code: i for i, code in enumerate(TARGET_COUNTRIES)}

# --- DATASET CLASS ---
class WebDatasetStream(IterableDataset):
    def __init__(self, urls, transform, country_map):
        self.urls = urls
        self.transform = transform
        self.country_map = country_map

    def __iter__(self):
        # 1. Create the pipeline
        # .shuffle(1000) keeps 1000 images in memory to randomize the stream
        dataset = (
            wds.WebDataset(self.urls, resampled=True, handler=wds.warn_and_continue)
            .shuffle(1000)
            .decode("pil")
            .to_tuple("jpg", "json")
        )

        for img, meta in dataset:
            try:
                # 2. Get Country (Safely)
                country = meta.get('country')
                
                # Filter: Skip if country is not in our specific list
                if not country or country not in self.country_map:
                    continue
                
                # 3. Transform Image
                # Convert to RGB to ensure 3 channels (skips greyscale issues)
                img_tensor = self.transform(img.convert("RGB"))
                label = self.country_map[country]
                
                yield img_tensor, label
                
            except Exception as e:
                # If a specific image fails, just move to the next one
                continue

# --- MAIN ---
# 1. Initialize Model
model = GeoguessrModel(num_classes=len(TARGET_COUNTRIES))
model = model.to(DEVICE)

# 2. Get the correct image size/stats automatically
data_config = model.get_config()
transforms = timm.data.create_transform(**data_config, is_training=True)

# 3. GENERATE CORRECT URLS (0000.tar to 0489.tar)
print("Generating URLs for OSV5M...")

# Base URL pattern
base_url = "https://huggingface.co/datasets/osv5m/osv5m-wds/resolve/main/train/{:04d}.tar"

# Generate list: 0000.tar ... 0489.tar
urls = [base_url.format(i) for i in range(490)]

print(f"First URL: {urls[0]}")
print(f"Last URL:  {urls[-1]}")
print(f"Total Shards: {len(urls)}")

# 4. Connect Dataset
train_dataset = WebDatasetStream(urls, transforms, country_to_idx)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

# 5. Training Loop
optimizer = optim.Adam(model.head.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print("\n--- STARTING TRAINING ---")
model.train()

for step, (imgs, labels) in enumerate(train_loader):
    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
    
    optimizer.zero_grad()
    outputs = model(imgs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # Print stats every 10 steps
    if (step + 1) % 10 == 0:
        print(f"Step {step+1}: Loss = {loss.item():.4f}")
    
    # Stop after set number of steps
    if step >= STEPS:
        break

# Save Result
print("\nSaving model...")
torch.save(model.state_dict(), "geoguessr.pth")
print("Done!")