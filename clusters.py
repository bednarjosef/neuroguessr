import pandas as pd
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from huggingface_hub import hf_hub_download
import os

TARGET_COUNTRIES = [
    'AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG', 'KH', 'CA', 'CL', 'CO', 'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI', 'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS', 'IN', 'ID', 'IE', 'IL', 'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY', 'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA', 'PE', 'PH', 'PL', 'PT', 'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS', 'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE', 'CH', 'TW', 'TH', 'TR', 'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN',
]

def latlon_to_cartesian(lat, lon):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    if isinstance(lat, np.ndarray):
        return np.stack([x, y, z], axis=1)
    return x, y, z

class ClusterManager:
    def __init__(self, n_clusters=200, cache_dir="./cluster_cache"):
        self.n_clusters = n_clusters
        self.cache_dir = cache_dir
        self.centers_path = os.path.join(cache_dir, f"centers_{n_clusters}.npy")
        self.weights_path = os.path.join(cache_dir, f"weights_{n_clusters}.pt")
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def generate(self):
        print(f"--- GENERATING {self.n_clusters} CLUSTERS ---")
        print("Downloading/Loading OSV5M metadata...")
        try:
            csv_path = hf_hub_download(repo_id="osv5m/osv5m", filename="train.csv", repo_type="dataset")
        except Exception as e:
            raise ConnectionError(f"Could not download dataset metadata: {e}")

        df = pd.read_csv(csv_path, usecols=['country', 'latitude', 'longitude'])
        
        print(f"Filtering for {len(TARGET_COUNTRIES)} target countries...")
        df = df[df['country'].isin(TARGET_COUNTRIES)].dropna()
        print(f"Dataset subset size: {len(df):,} images")

        print("Running MiniBatch K-Means (This may take a minute)...")
        coords_3d = latlon_to_cartesian(df['latitude'].values, df['longitude'].values)
        
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=4096)
        labels = kmeans.fit_predict(coords_3d)
        centers = kmeans.cluster_centers_

        print("Calculating Loss Weights for Imbalance...")
        counts = np.bincount(labels, minlength=self.n_clusters)
        safe_counts = np.maximum(counts, 1) # Prevent div by zero
        
        weights = len(df) / (self.n_clusters * safe_counts)
        weights = weights / np.median(weights)
        
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        print(f"Saving clusters and weights to {self.cache_dir}...")
        np.save(self.centers_path, centers)
        torch.save(weights_tensor, self.weights_path)
        
        return centers, weights_tensor

    def load(self, device='cpu'):
        if not os.path.exists(self.centers_path) or not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Clusters not found in {self.cache_dir}. Run generate() first.")
            
        print(f"Loading clusters from {self.centers_path}...")
        centers = np.load(self.centers_path)
        weights = torch.load(self.weights_path, map_location=device)
        
        return centers, weights
    
    def generate_soft_targets(self, sigma_km=500.0):
        print(f"Generating Distance-Based Soft Targets (Sigma={sigma_km}km)...")
        centers, _ = self.load(device='cpu') # Shape (N, 3)
        
        # 1. Convert Centers to Lat/Lon for Haversine
        # (Reusing your math logic)
        z = centers[:, 2]
        y = centers[:, 1]
        x = centers[:, 0]
        lats = np.degrees(np.arcsin(z))
        lons = np.degrees(np.arctan2(y, x))
        
        # 2. Compute Pairwise Haversine Distances (N x N)
        # Broadcasting magic to do it fast without loops
        lat_rad = np.radians(lats)
        lon_rad = np.radians(lons)
        
        # Shape: (N, 1) and (1, N)
        lat1 = lat_rad[:, None]
        lat2 = lat_rad[None, :]
        lon1 = lon_rad[:, None]
        lon2 = lon_rad[None, :]
        
        dphi = lat2 - lat1
        dlambda = lon2 - lon1
        
        a = np.sin(dphi/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        dists_km = 6371 * c
        
        # 3. Apply Gaussian Kernel
        # P(x) = exp(-dist^2 / (2 * sigma^2))
        targets = np.exp(-(dists_km**2) / (2 * sigma_km**2))
        
        # 4. Normalize so rows sum to 1.0 (Probability Distribution)
        # This ensures it works with CrossEntropyLoss
        sums = targets.sum(axis=1, keepdims=True)
        soft_targets = targets / sums
        
        # Convert to Tensor (N, N)
        return torch.tensor(soft_targets, dtype=torch.float32)

    def get_closest_label(self, lat, lon, centers_tensor):
        """
        Vectorized helper to find the closest cluster ID for a batch of lat/lons.
        Used inside the Dataset class.
        """
        # Note: Logic usually implemented inside Dataset __iter__ for speed, 
        # but provided here if needed.
        pass