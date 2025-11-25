import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds
from model import GeoguessrModel
import glob
import os
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
LOCAL_DATA_DIR = "./osv5m_local_data" 
MICRO_BATCH_SIZE = 128
ACCUM_STEPS = 6
LEARNING_RATE = 5e-4     
STEPS = 12
DEVICE = "cuda"
NUM_WORKERS = 12
NUM_CLUSTERS = 200 # The requested 200 Geocells

# The specific list you requested
TARGET_COUNTRIES = [
    'AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG', 'KH', 'CA', 'CL', 'CO', 'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI', 'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS', 'IN', 'ID', 'IE', 'IL', 'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY', 'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA', 'PE', 'PH', 'PL', 'PT', 'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS', 'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE', 'CH', 'TW', 'TH', 'TR', 'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN',
]

# --- HELPER: 3D COORDINATES ---
def latlon_to_cartesian(lat, lon):
    """Converts lat/lon to 3D cartesian (x, y, z) unit vectors."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=1)

# --- PRE-PROCESSING: CLUSTER GENERATION ---
def generate_clusters_and_weights():
    """
    Downloads metadata, filters for target countries, generates K-Means clusters,
    and calculates loss weights for imbalance handling.
    """
    print("--- STEP 1: LOADING METADATA & GENERATING CLUSTERS ---")
    repo_id = "osv5m/osv5m"
    filename = "train.csv"
    
    try:
        csv_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    except Exception as e:
        print(f"Error downloading metadata: {e}")
        exit()

    print("Reading CSV and filtering...")
    df = pd.read_csv(csv_path, usecols=['country', 'latitude', 'longitude'])
    df = df[df['country'].isin(TARGET_COUNTRIES)].dropna()
    
    print(f"Training on subset of {len(df):,} images from {len(TARGET_COUNTRIES)} countries.")

    # 1. Run Clustering
    print(f"Running K-Means (K={NUM_CLUSTERS})...")
    coords_3d = latlon_to_cartesian(df['latitude'].values, df['longitude'].values)
    kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, random_state=42, batch_size=4096)
    labels = kmeans.fit_predict(coords_3d)
    
    # 2. Calculate Weights (Replacement for WeightedRandomSampler)
    # We calculate the inverse frequency of each cluster.
    # Rare clusters get high weights, Common clusters get low weights.
    print("Calculating Class Balance Weights...")
    counts = np.bincount(labels, minlength=NUM_CLUSTERS)
    
    # Avoid div by zero
    safe_counts = np.maximum(counts, 1) 
    
    # Formula: Total / (Classes * Count)
    weights = len(df) / (NUM_CLUSTERS * safe_counts)
    
    # Normalize so median is 1.0 (keeps learning rate stable)
    weights = weights / np.median(weights)
    
    print(f"Weight Stats -> Min: {weights.min():.2f}, Max: {weights.max():.2f}")
    
    return kmeans.cluster_centers_, torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# --- DATASET ---
class LocalWebDatasetGeocells(IterableDataset):
    def __init__(self, tar_paths, transform, target_countries, cluster_centers):
        self.tar_paths = tar_paths
        self.transform = transform
        self.target_countries = set(target_countries)
        
        # We store the cluster centers in the dataset to label on-the-fly
        self.cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32)

    def get_closest_cluster(self, lat, lon):
        """
        Finds the nearest cluster center to the image.
        Done via Dot Product (Cosine Similarity) since vectors are normalized.
        """
        # Convert single point to 3D
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        
        point = torch.tensor([x, y, z], dtype=torch.float32)
        
        # Calculate distance (Euclidean squared is faster and sufficient for finding min)
        # dist^2 = (x1-x2)^2 + ...
        dists = torch.sum((self.cluster_centers - point)**2, dim=1)
        return torch.argmin(dists).item()

    def __iter__(self):
        dataset = (
            wds.WebDataset(self.tar_paths, resampled=True, shardshuffle=True, handler=wds.warn_and_continue)
            .shuffle(5000) 
            .decode("pil")
            .to_tuple("jpg", "json")
        )

        for img, meta in dataset:
            try:
                # 1. Filter Country
                country = meta.get('country')
                if not country or country not in self.target_countries:
                    continue
                
                # 2. Get Coordinates
                lat = meta.get('latitude')
                lon = meta.get('longitude')
                if lat is None or lon is None:
                    continue

                # 3. Assign Label (Find closest cluster)
                label = self.get_closest_cluster(lat, lon)

                # 4. Transform Image
                img_tensor = self.transform(img.convert("RGB"))
                
                yield img_tensor, label
            except Exception:
                continue

# --- MAIN ---
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high') 
    
    # 1. Generate the Map (Clusters) and Weights
    cluster_centers, loss_weights = generate_clusters_and_weights()
    
    # 2. Model Setup (Num Classes = Num Clusters)
    print(f"Initializing Model with {NUM_CLUSTERS} output classes...")
    model = GeoguessrModel(num_classes=NUM_CLUSTERS)
    model = model.to(DEVICE)
    model.head = torch.compile(model.head) 

    # 3. Find Local Files
    search_path = os.path.join(LOCAL_DATA_DIR, "train", "*.tar")
    tar_files = glob.glob(search_path)
    if not tar_files:
        tar_files = glob.glob(os.path.join(LOCAL_DATA_DIR, "*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No .tar files found in {LOCAL_DATA_DIR}")

    print(f"Found {len(tar_files)} local tar shards.")

    # 4. Data Loaders
    data_config = model.get_config()
    transforms = timm.data.create_transform(**data_config, is_training=True)

    # Pass the cluster centers to the dataset so it can label images
    train_dataset = LocalWebDatasetGeocells(tar_files, transforms, TARGET_COUNTRIES, cluster_centers)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=MICRO_BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # 5. Optimizer & Loop
    optimizer = optim.AdamW(model.head.parameters(), lr=LEARNING_RATE)
    
    # KEY CHANGE: Pass the calculated weights to the loss function
    # This replaces the WeightedRandomSampler
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    scaler = torch.amp.GradScaler('cuda')

    print(f"--- TRAINING START (Geocell Classification) ---")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True) # Labels are now 0-199
        
        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels) / ACCUM_STEPS 
        
        scaler.scale(loss).backward()
        
        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Simple progress log
            print(f"Step {step+1}: Loss = {loss.item() * ACCUM_STEPS:.4f}")
        
        if step >= (STEPS * ACCUM_STEPS):
            break

    torch.save(model.state_dict(), "geoguessr_geocells_200.pth")
    # Save the clusters too, otherwise the model is useless!
    np.save("cluster_centers_200.npy", cluster_centers)
    print("Training Complete. Model and Clusters saved.")