import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds
from model import GeoguessrModel
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np

MODEL_PATH = "geoguessr_optimized.pth"
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_COUNTRIES = ["FR", "US", "JP", "GB", "AU", "BR", "ZA", "KR", "ES", "IT"]
country_to_idx = {code: i for i, code in enumerate(TARGET_COUNTRIES)}
idx_to_country = {i: code for code, i in country_to_idx.items()}

class WebDatasetEval(IterableDataset):
    def __init__(self, urls, transform, country_map):
        self.urls = urls
        self.transform = transform
        self.country_map = country_map

    def __iter__(self):
        dataset = (
            wds.WebDataset(self.urls, resampled=False, handler=wds.warn_and_continue)
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

if __name__ == "__main__":
    print(f"Evaluating on device: {DEVICE}")

    model = GeoguessrModel(num_classes=len(TARGET_COUNTRIES))
    
    # Load weights
    # We use strict=False in case you compiled the model (which adds _orig_mod prefixes)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    # Fix for torch.compile keys if needed
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model = model.to(DEVICE)
    model.eval()

    # 2. Setup Data
    data_config = model.get_config()
    transforms = timm.data.create_transform(**data_config, is_training=False)

    print("Generating Validation URLs...")
    base_url = "https://huggingface.co/datasets/osv5m/osv5m-wds/resolve/main/test/{:04d}.tar"
    urls = [base_url.format(i) for i in range(49)] # 0 to 48

    val_dataset = WebDatasetEval(urls, transforms, country_to_idx)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

    print("Starting Inference...")
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store on CPU for metrics calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (i+1) % 50 == 0:
                print(f"Processed {len(all_preds)} images...")

    print("\n--- EVALUATION RESULTS ---")
    
    avg_loss = total_loss / len(val_loader)
    print(f"Eval Loss: {avg_loss:.4f}")

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro):    {recall:.4f}")
    print(f"F1 Score (Macro):  {f1:.4f}")

    print("\n--- DETAILED REPORT ---")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=TARGET_COUNTRIES, 
        digits=3,
        zero_division=0
    ))
