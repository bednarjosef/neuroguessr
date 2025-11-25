import torch
import torch.nn as nn
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from clusters import ClusterManager, latlon_to_cartesian

class LocalValDataset(Dataset):
    def __init__(self, root_dir, transform, cluster_manager):
        self.samples = []
        self.transform = transform
        self.cluster_manager = cluster_manager
        
        # Walk through the cache directory
        for country in os.listdir(root_dir):
            country_dir = os.path.join(root_dir, country)
            if not os.path.isdir(country_dir): continue
            
            for fname in os.listdir(country_dir):
                if fname.endswith(".jpg"):
                    img_path = os.path.join(country_dir, fname)
                    meta_path = img_path.replace(".jpg", ".json")
                    
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    self.samples.append({
                        'img_path': img_path,
                        'lat': meta['latitude'],
                        'lon': meta['longitude'],
                        'country': country
                    })
        
        print(f"Evaluator loaded {len(self.samples)} images from disk.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item['img_path']).convert('RGB')
        img = self.transform(img)
        return img, item['lat'], item['lon'], item['country']

class Evaluator:
    def __init__(self, val_dir, model_config, n_clusters, batch_size=128, device="cuda"):
        self.device = device
        
        # Load Clusters (Centers only)
        self.cm = ClusterManager(n_clusters=n_clusters) 
        self.cm.centers, _ = self.cm.load(device='cpu') 
        
        # Setup Data
        import timm
        transforms = timm.data.create_transform(**model_config, is_training=False)
        self.dataset = LocalValDataset(val_dir, transforms, self.cm)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        
        # Pre-load centers to GPU
        self.centers_gpu = torch.tensor(self.cm.centers, device=device, dtype=torch.float32)

    def haversine(self, lat1, lon1, lat2, lon2):
        # Vectorized Haversine Formula
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def calculate_geoscore(self, km_errors):
        # Standard GeoGuessr Formula: 5000 * e^(-distance / 2000)
        # 0km error = 5000 pts. 2000km error = ~1800 pts.
        return 5000 * np.exp(-np.array(km_errors) / 2000)

    def run(self, model):
        model.eval()
        all_dists = []
        
        print("\n--- RUNNING EVALUATION ---")
        with torch.no_grad():
            for imgs, true_lats, true_lons, _ in self.loader:
                imgs = imgs.to(self.device)
                
                # Inference
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                
                # Decode Predictions (Cluster ID -> Lat/Lon)
                pred_centers = self.centers_gpu[preds]
                z = pred_centers[:, 2]
                y = pred_centers[:, 1]
                x = pred_centers[:, 0]
                pred_lats = torch.rad2deg(torch.asin(z)).cpu().numpy()
                pred_lons = torch.rad2deg(torch.atan2(y, x)).cpu().numpy()
                
                # Calculate Distances
                dists = self.haversine(true_lats.numpy(), true_lons.numpy(), pred_lats, pred_lons)
                all_dists.extend(dists)

        # --- CALCULATE METRICS ---
        mean_km = np.mean(all_dists)
        median_km = np.median(all_dists)
        geo_scores = self.calculate_geoscore(all_dists)
        mean_score = np.mean(geo_scores)

        print(f"Eval -> Mean: {mean_km:.1f}km | Median: {median_km:.1f}km | Score: {mean_score:.0f}")
        
        model.train()
        
        # Return dict for WandB
        return {
            "val/mean_km": mean_km,
            "val/median_km": median_km,
            "val/geo_score": mean_score
        }