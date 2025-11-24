import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds
from model import GeoguessrModel
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = "geoguessr_local_1.pth"
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Target Countries (Must match training exactly!)
TARGET_COUNTRIES = [
    'AD', 'AE', 'AF', 'AG', 'AI', 'AL', 'AM', 'AO', 'AR', 'AT', 'AU', 'AW', 
    'AX', 'AZ', 'BA', 'BB', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BM', 
    'BN', 'BO', 'BQ', 'BR', 'BS', 'BT', 'BW', 'BY', 'BZ', 'CA', 'CD', 'CF', 
    'CG', 'CH', 'CI', 'CL', 'CM', 'CN', 'CO', 'CR', 'CU', 'CV', 'CW', 'CX', 
    'CY', 'CZ', 'DE', 'DJ', 'DK', 'DM', 'DO', 'DZ', 'EC', 'EE', 'EG', 'ER', 
    'ES', 'ET', 'FI', 'FJ', 'FK', 'FM', 'FO', 'FR', 'GA', 'GB', 'GD', 'GE', 
    'GF', 'GG', 'GH', 'GI', 'GL', 'GM', 'GN', 'GP', 'GQ', 'GR', 'GT', 'GW', 
    'GY', 'HK', 'HN', 'HR', 'HT', 'HU', 'ID', 'IE', 'IL', 'IM', 'IN', 'IQ', 
    'IR', 'IS', 'IT', 'JE', 'JM', 'JO', 'JP', 'KE', 'KG', 'KH', 'KI', 'KM', 
    'KN', 'KP', 'KR', 'KW', 'KZ', 'LA', 'LB', 'LC', 'LI', 'LK', 'LR', 'LS', 
    'LT', 'LU', 'LV', 'LY', 'MA', 'MC', 'MD', 'ME', 'MF', 'MG', 'MK', 'ML', 
    'MM', 'MN', 'MO', 'MQ', 'MR', 'MS', 'MT', 'MU', 'MV', 'MW', 'MX', 'MY', 
    'MZ', 'NC', 'NE', 'NG', 'NI', 'NL', 'NO', 'NP', 'NZ', 'OM', 'PA', 'PE', 
    'PF', 'PG', 'PH', 'PK', 'PL', 'PR', 'PS', 'PT', 'PW', 'PY', 'QA', 'RE', 
    'RO', 'RS', 'RU', 'RW', 'SA', 'SB', 'SD', 'SE', 'SG', 'SI', 'SJ', 'SK', 
    'SL', 'SM', 'SN', 'SO', 'SR', 'SS', 'ST', 'SV', 'SX', 'SY', 'SZ', 'TC', 
    'TD', 'TG', 'TH', 'TJ', 'TL', 'TM', 'TN', 'TR', 'TT', 'TW', 'TZ', 'UA', 
    'UG', 'US', 'UY', 'UZ', 'VA', 'VC', 'VE', 'VG', 'VI', 'VN', 'VU', 'WS', 
    'XK', 'YE', 'YT', 'ZA', 'ZM', 'ZW', 
]

TARGET_COUNTRIES = ['AU', 'BR', 'CA', 'CZ', 'FR', 'ID', 'IN', 'JP', 'MX', 'RU']

country_to_idx = {code: i for i, code in enumerate(TARGET_COUNTRIES)}
idx_to_country = {i: code for code, i in country_to_idx.items()}

# --- DATASET CLASS (No Shuffle for Eval) ---
class WebDatasetEval(IterableDataset):
    def __init__(self, urls, transform, country_map):
        self.urls = urls
        self.transform = transform
        self.country_map = country_map

    def __iter__(self):
        # NOTE: No shardshuffle=True and no .shuffle() buffer
        # We want deterministic evaluation
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

# --- MAIN EVALUATION ---
if __name__ == "__main__":
    print(f"Evaluating on device: {DEVICE}")

    # 1. Load Model
    model = GeoguessrModel(num_classes=len(TARGET_COUNTRIES))
    
    # Load weights safely
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        # Clean up torch.compile keys if they exist
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit()
    
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

    # 3. Run Inference
    print("Starting Inference...")
    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1  # Increment batch counter
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (i+1) % 50 == 0:
                print(f"Processed {len(all_preds)} images...")

    # 4. Calculate Metrics
    print("\n--- EVALUATION RESULTS ---")
    
    # FIX: Divide by our manual counter, not len(val_loader)
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"Eval Loss: {avg_loss:.4f}")
    else:
        print("Eval Loss: N/A (No data found)")

    # Metrics
    if len(all_labels) > 0:
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
    else:
        print("No valid images found for the target countries in the validation set.")