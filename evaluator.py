import torch
import torch.nn as nn
import os
import json
import numpy as np
import datetime
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from clusters import ClusterManager, latlon_to_cartesian

class LocalValDataset(Dataset):
    def __init__(self, root_dir, transform, cluster_manager):
        self.samples = []
        self.transform = transform
        self.cm = cluster_manager
        
        # Pre-load centers
        self.centers = torch.tensor(cluster_manager.centers, dtype=torch.float32)
        
        for country in os.listdir(root_dir):
            country_dir = os.path.join(root_dir, country)
            if not os.path.isdir(country_dir): continue
            
            for fname in os.listdir(country_dir):
                if fname.endswith(".jpg"):
                    img_path = os.path.join(country_dir, fname)
                    meta_path = img_path.replace(".jpg", ".json")
                    
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    # EXTRACT METADATA (Default to -1 if missing)
                    clim = int(meta.get('climate', -1))
                    land = int(meta.get('land_cover', -1))
                    soil = int(meta.get('soil', -1))
                    
                    # Parse Month
                    ts = meta.get('captured_at')
                    if ts:
                        month = datetime.datetime.fromtimestamp(ts / 1000.0).month - 1
                    else:
                        month = -1
                    
                    self.samples.append({
                        'img_path': img_path,
                        'lat': meta['latitude'],
                        'lon': meta['longitude'],
                        'clim': clim,
                        'land': land,
                        'soil': soil,
                        'month': month
                    })
        print(f"Evaluator loaded {len(self.samples)} images.")

    def __len__(self): return len(self.samples)
    
    def get_closest_label(self, lat, lon):
        x, y, z = latlon_to_cartesian(lat, lon)
        point = torch.tensor([x, y, z], dtype=torch.float32)
        return torch.argmin(torch.sum((self.centers - point)**2, dim=1)).item()

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item['img_path']).convert('RGB')
        img = self.transform(img)
        
        # Calc ground truth loc
        label_loc = self.get_closest_label(item['lat'], item['lon'])
        
        return (img, label_loc, item['lat'], item['lon'], 
                item['clim'], item['land'], item['soil'], item['month'])

class Evaluator:
    def __init__(self, val_dir, model_config, num_clusters, batch_size=128, device="cuda"):
        self.device = device
        self.cm = ClusterManager(n_clusters=num_clusters)
        self.cm.centers, _ = self.cm.load(device='cpu')
        
        import timm
        transforms = timm.data.create_transform(**model_config, is_training=False)
        self.dataset = LocalValDataset(val_dir, transforms, self.cm)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, num_workers=4)
        self.centers_gpu = torch.tensor(self.cm.centers, device=device, dtype=torch.float32)

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def run(self, model):
        model.eval()
        all_dists = []
        
        # Track accuracy for all heads
        stats = {k: {'correct': 0, 'total': 0} for k in ['top1', 'top5', 'clim', 'land', 'soil', 'month']}
        
        print("\n--- RUNNING EVALUATION ---")
        with torch.no_grad():
            for batch in self.loader:
                # Unpack (8 items!)
                imgs = batch[0].to(self.device)
                lbl_loc = batch[1].to(self.device)
                true_lats = batch[2]
                true_lons = batch[3]
                lbl_clim = batch[4].to(self.device)
                lbl_land = batch[5].to(self.device)
                lbl_soil = batch[6].to(self.device)
                lbl_month = batch[7].to(self.device)
                
                # Forward
                out_loc, out_clim, out_land, out_soil, out_month = model(imgs)
                
                # --- METRICS CALC ---
                
                # 1. Location Acc
                _, preds_loc = torch.max(out_loc, 1)
                stats['top1']['correct'] += (preds_loc == lbl_loc).sum().item()
                stats['top1']['total'] += lbl_loc.size(0)
                
                _, top5 = out_loc.topk(5, 1, True, True)
                stats['top5']['correct'] += (top5 == lbl_loc.unsqueeze(1)).sum().item()
                stats['top5']['total'] += lbl_loc.size(0)
                
                # 2. Aux Acc (Handle -1 ignore index)
                def update_acc(name, outputs, labels):
                    mask = labels != -1
                    if mask.sum() > 0:
                        _, preds = torch.max(outputs, 1)
                        stats[name]['correct'] += (preds[mask] == labels[mask]).sum().item()
                        stats[name]['total'] += mask.sum().item()

                update_acc('clim', out_clim, lbl_clim)
                update_acc('land', out_land, lbl_land)
                update_acc('soil', out_soil, lbl_soil)
                update_acc('month', out_month, lbl_month)

                # 3. Distances
                pred_centers = self.centers_gpu[preds_loc]
                z, y, x = pred_centers[:, 2], pred_centers[:, 1], pred_centers[:, 0]
                pred_lats = torch.rad2deg(torch.asin(z)).cpu().numpy()
                pred_lons = torch.rad2deg(torch.atan2(y, x)).cpu().numpy()
                dists = self.haversine(true_lats.numpy(), true_lons.numpy(), pred_lats, pred_lons)
                all_dists.extend(dists)

        # Summarize
        mean_km = np.mean(all_dists)
        med_km = np.median(all_dists)
        
        # Calculate percentages
        results = {
            "val/mean_km": mean_km,
            "val/median_km": med_km,
            "val/geo_score": np.mean(5000 * np.exp(-np.array(all_dists)/2000))
        }
        
        for k, v in stats.items():
            acc = (v['correct'] / v['total'] * 100) if v['total'] > 0 else 0
            results[f"val/acc_{k}"] = acc
            print(f"{k.upper()}: {acc:.2f}%")

        print(f"Mean: {mean_km:.0f}km | Med: {med_km:.0f}km")
        
        model.train()
        return results