import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds
from model import GeoguessrModel
import os

# CONFIG
BATCH_SIZE = 512 
LEARNING_RATE = 5e-4
STEPS = 100
DEVICE = "cuda"
NUM_WORKERS = 12

# Target Countries
TARGET_COUNTRIES = ["FR", "US", "JP", "GB", "AU", "BR", "ZA", "KR", "ES", "IT"]
country_to_idx = {code: i for i, code in enumerate(TARGET_COUNTRIES)}

class WebDatasetStream(IterableDataset):
    def __init__(self, urls, transform, country_map):
        self.urls = urls
        self.transform = transform
        self.country_map = country_map

    def __iter__(self):
        dataset = (
            wds.WebDataset(self.urls, resampled=True, shardshuffle=True, handler=wds.warn_and_continue)
            .shuffle(5000)
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


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    model = GeoguessrModel(num_classes=len(TARGET_COUNTRIES))
    model = model.to(DEVICE)

    print("Compiling model (this takes ~1 minute at start)...")
    model.head = torch.compile(model.head)

    data_config = model.get_config()
    transforms = timm.data.create_transform(**data_config, is_training=True)

    base_url = "https://huggingface.co/datasets/osv5m/osv5m-wds/resolve/main/train/{:04d}.tar"
    urls = [base_url.format(i) for i in range(490)]

    train_dataset = WebDatasetStream(urls, transforms, country_to_idx)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        prefetch_factor=2
    )

    optimizer = optim.AdamW(model.head.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    scaler = torch.amp.GradScaler('cuda')

    print("\n--- STARTING TRAINING (OPTIMIZED) ---")
    model.train()

    for step, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % 1 == 0:
            print(f"Step {step}: Loss = {loss.item():.8f}")
        
        if step >= STEPS:
            break

    print("\nSaving optimized model...")
    torch.save(model.state_dict(), "geoguessr_optimized.pth")
    print("Done!")
    