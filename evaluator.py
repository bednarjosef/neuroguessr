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
        
        # Pre-load cluster centers to CPU for fast label calculation
        self.centers = torch.tensor(cluster_manager.centers, dtype=torch.float32)
        
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

    def get_closest_cluster_label(self, lat, lon):
        # Calculate 3D position
        x, y, z = latlon_to_cartesian(lat, lon)
        point = torch.tensor([x, y, z], dtype=torch.float32)
        
        # Find nearest center (Ground Truth Class)
        dists = torch.sum((self.centers - point)**2, dim=1)
        return torch.argmin(dists).item()

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item['img_path']).convert('RGB')
        img = self.transform(img)
        
        # Compute Ground Truth Label on the fly
        label = self.get_closest_cluster_label(item['lat'], item['lon'])
        
        return img, label, item['lat'], item['lon'], item['country']

class Evaluator:
    def __init__(self, val_dir, model_config, num_clusters, batch_size=128, device="cuda"):
        self.device = device
        
        # Load Clusters (Centers only)
        self.cm = ClusterManager(n_clusters=num_clusters) # Will load default N from cache
        self.cm.centers, _ = self.cm.load(device='cpu') 
        
        # Setup Data
        import timm
        transforms = timm.data.create_transform(**model_config, is_training=False)
        self.dataset = LocalValDataset(val_dir, transforms, self.cm)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        
        # Pre-load centers to GPU for distance calc
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
        # 5000 * e^(-distance / 2000)
        return 5000 * np.exp(-np.array(km_errors) / 2000)

    def run(self, model):
        model.eval()
        all_dists = []
        
        # Counters for Accuracy
        correct_top1 = 0
        correct_top5 = 0
        total_samples = 0
        
        print("\n--- RUNNING EVALUATION ---")
        with torch.no_grad():
            for imgs, labels, true_lats, true_lons, _ in self.loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                # Inference
                outputs = model(imgs) # shape: [Batch, Num_Classes]
                
                # --- ACCURACY CALCS ---
                # Top-1
                _, preds = torch.max(outputs, 1)
                correct_top1 += (preds == labels).sum().item()
                
                # Top-5
                # Get the indices of the top 5 predictions
                # top5_preds shape: [Batch, 5]
                _, top5_preds = outputs.topk(5, 1, True, True)
                # Check if true label is in those 5 columns
                correct_top5 += (top5_preds == labels.unsqueeze(1)).sum().item()
                
                total_samples += labels.size(0)
                
                # --- DISTANCE CALCS ---
                # Decode Predictions (Cluster ID -> Lat/Lon) using Top-1
                pred_centers = self.centers_gpu[preds]
                z = pred_centers[:, 2]
                y = pred_centers[:, 1]
                x = pred_centers[:, 0]
                pred_lats = torch.rad2deg(torch.asin(z)).cpu().numpy()
                pred_lons = torch.rad2deg(torch.atan2(y, x)).cpu().numpy()
                
                # Calculate Distances
                dists = self.haversine(true_lats.numpy(), true_lons.numpy(), pred_lats, pred_lons)
                all_dists.extend(dists)

        # --- METRICS ---
        mean_km = np.mean(all_dists)
        median_km = np.median(all_dists)
        geo_scores = self.calculate_geoscore(all_dists)
        mean_score = np.mean(geo_scores)
        
        acc_top1 = (correct_top1 / total_samples) * 100
        acc_top5 = (correct_top5 / total_samples) * 100

        print(f"Eval -> Top1: {acc_top1:.2f}% | Top5: {acc_top5:.2f}% | Score: {mean_score:.0f} |  Mean: {mean_km:.0f}km | Med: {median_km:.0f}km")
        
        model.train()
        return {
            "val/acc_top1": acc_top1,
            "val/acc_top5": acc_top5,
            "val/geo_score": mean_score,
            "val/mean_km": mean_km,
            "val/median_km": median_km,
        }