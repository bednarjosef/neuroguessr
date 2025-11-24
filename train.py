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

weights = torch.tensor([
    11.5021, 0.1723, 9.8715, 4.0007, 368.0674, 0.4679, 3.2655, 1.3874, 0.0620, 
    0.0656, 0.0159, 2576.4720, 3.7340, 1.0844, 0.3451, 2.7007, 0.3038, 0.1045, 
    4.3085, 0.1917, 0.3007, 7.7839, 2.2482, 51.5294, 0.9349, 0.1466, 2576.4720, 
    0.0183, 6.5559, 62.8408, 1.4266, 0.1025, 1.9184, 0.0306, 0.8900, 28.3129, 
    40.8964, 0.1294, 2.2078, 0.1127, 1.3226, 0.3321, 0.0985, 0.4084, 1.7806, 
    6.5559, 23.2115, 644.1180, 0.1766, 0.1263, 0.0095, 135.6038, 0.0557, 2.9959, 
    0.4850, 0.3497, 0.2149, 0.0994, 0.5162, 2576.4720, 0.0267, 0.4053, 0.0560, 
    8.4752, 52.5811, 3.4630, 2.0303, 0.0106, 198.1902, 0.0187, 27.4093, 0.5861, 
    4.6423, 12.6920, 0.2909, 26.5616, 73.6135, 13.3496, 1.9519, 1.6411, 858.8240, 
    0.0613, 0.3350, 10.4735, 4.7712, 0.6988, 0.7393, 0.1169, 2.5185, 0.0621, 
    0.0704, 0.0706, 0.3716, 2.8282, 0.0509, 2.3659, 0.3285, 0.3370, 0.0224, 
    7.1768, 42.2372, 0.4046, 0.0123, 1.6043, 1.1616, 5.8690, 322.0590, 31.4204, 
    9.6137, 1288.2360, 0.7770, 0.2376, 0.3415, 0.6398, 2.2541, 1.7843, 12.6298, 
    0.5553, 1.8931, 0.5778, 0.2002, 0.8737, 0.1052, 22.4041, 0.1246, 39.6380, 
    0.1689, 1.6103, 644.1180, 2.0745, 0.4316, 0.8192, 0.6002, 1.1380, 11.2021, 
    1.9299, 11.7647, 37.8893, 1.8106, 1.0033, 128.8236, 6.0910, 0.0482, 0.2793, 
    0.6469, 19.6677, 15.2454, 0.2584, 0.3180, 0.0519, 0.0922, 0.5937, 0.0566, 
    0.2445, 0.5838, 0.1756, 1288.2360, 1.5165, 0.2610, 0.0402, 0.0243, 6.6404, 
    1.0327, 0.0500, 429.4120, 0.2167, 0.5093, 0.6732, 0.0454, 0.1265, 0.0116, 
    0.5889, 0.7855, 4.4653, 122.6891, 0.0195, 0.5491, 0.2800, 18.6701, 0.1323, 
    0.3332, 30.6723, 0.9967, 286.2747, 4.1158, 368.0674, 50.5191, 0.8089, 1288.2360, 
    30.3114, 15.6150, 33.4607, 85.8824, 1.3793, 0.0345, 1.7222, 0.6475, 0.4870, 
    0.1883, 0.0587, 2.0464, 0.0807, 0.4904, 3.1848, 0.3184, 0.0021, 0.4691, 
    0.5748, 10.3890, 8.8539, 234.2247, 14.9795, 322.0590, 0.2117, 7.8551, 8.6459, 
    1.1923, 4.3302, 2.5637, 0.0482, 0.2580, 3.8170, 
]).to(DEVICE)

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
    criterion = nn.CrossEntropyLoss(weight=weights)
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
