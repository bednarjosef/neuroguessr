import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds
from model import GeoguessrModel
import glob
import os

# --- CONFIGURATION ---
LOCAL_DATA_DIR = "./osv5m_local_data" # Must match where you downloaded
MICRO_BATCH_SIZE = 128
ACCUM_STEPS = 6
LEARNING_RATE = 5e-4    
STEPS = 1000 
DEVICE = "cuda"
# Higher workers now because SSDs can handle parallel reads easily
NUM_WORKERS = 12

# Target Countries
TARGET_COUNTRIES = ["FR", "US", "JP", "GB", "BR", "ZA", "IT", "CZ"]
country_to_idx = {code: i for i, code in enumerate(TARGET_COUNTRIES)}

# --- DATASET ---
class LocalWebDatasetStream(IterableDataset):
    def __init__(self, tar_paths, transform, country_map):
        self.tar_paths = tar_paths
        self.transform = transform
        self.country_map = country_map

    def __iter__(self):
        # We pass the list of local filenames directly to WebDataset
        dataset = (
            wds.WebDataset(self.tar_paths, resampled=True, shardshuffle=True, handler=wds.warn_and_continue)
            .shuffle(10000) 
            .decode("pil")
            .to_tuple("jpg", "json")
        )

        for img, meta in dataset:
            try:
                country = meta.get('country')
                if not country or country not in self.country_map:
                    continue
                img_tensor = self.transform(img.convert("RGB"))
                label = self.country_map[country]
                yield img_tensor, label
            except Exception:
                continue

# --- MAIN ---
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high') 
    
    # 1. Model Setup
    model = GeoguessrModel(num_classes=len(TARGET_COUNTRIES))
    model = model.to(DEVICE)
    model.head = torch.compile(model.head) 

    # 2. Find Local Files
    # Look for the .tar files inside the local directory
    # Depending on how snapshot_download saves it, they might be in a 'train' subdir
    search_path = os.path.join(LOCAL_DATA_DIR, "train", "*.tar")
    tar_files = glob.glob(search_path)
    
    # Fallback search if folder structure is flat
    if not tar_files:
        tar_files = glob.glob(os.path.join(LOCAL_DATA_DIR, "*.tar"))

    if not tar_files:
        raise FileNotFoundError(f"No .tar files found in {LOCAL_DATA_DIR}. Did you run download.py?")

    print(f"Found {len(tar_files)} local tar shards. Training on SSD speed!")

    # 3. Data Loaders
    data_config = model.get_config()
    transforms = timm.data.create_transform(**data_config, is_training=True)

    train_dataset = LocalWebDatasetStream(tar_files, transforms, country_to_idx)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=MICRO_BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        prefetch_factor=4, # SSDs love prefetching
        persistent_workers=True
    )

    # 4. Optimizer & Loop
    optimizer = optim.AdamW(model.head.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    print(f"--- TRAINING START (LOCAL) ---")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels) / ACCUM_STEPS 
        
        scaler.scale(loss).backward()
        
        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            print(f"Step {step+1}: Effective Loss = {loss.item() * ACCUM_STEPS:.4f}")
        
        if step >= (STEPS * ACCUM_STEPS):
            break

    torch.save(model.state_dict(), "geoguessr_local.pth")
    print("Training Complete.")
