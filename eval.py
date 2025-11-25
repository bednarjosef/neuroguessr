import torch
import torch.nn as nn
import numpy as np
import math
from model import GeoguessrModel

# --- IMPORT MODULES ---
from clusters import ClusterManager
from dataset import create_dataloader

# --- CONFIGURATION ---
MODEL_PATH = "geoguessr_model_1.pth"
# Use a few shards from the official OSV5M test set
VAL_URLS = ["https://huggingface.co/datasets/osv5m/osv5m-wds/resolve/main/test/{:04d}.tar".format(i) for i in range(10)]
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLUSTERS = 200
NUM_WORKERS = 4

# --- MATH HELPERS ---
def cartesian_to_latlon(x, y, z):
    """Converts 3D cartesian coordinates back to Lat/Lon (degrees)."""
    # Ensure inputs are floats, not tensors, for numpy math
    if isinstance(x, torch.Tensor):
        x, y, z = x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy()
        
    lat_rad = np.arcsin(z)
    lon_rad = np.arctan2(y, x)
    return np.degrees(lat_rad), np.degrees(lon_rad)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance in km between two lat/lon points."""
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# --- MAIN ---
if __name__ == "__main__":
    print(f"--- STARTING EVALUATION (K={NUM_CLUSTERS}) ---")
    
    # 1. Load Knowledge (Cluster Centers)
    cluster_manager = ClusterManager(n_clusters=NUM_CLUSTERS)
    try:
        # We only need centers for eval. Weights are irrelevant here.
        cluster_centers, _ = cluster_manager.load(device='cpu') 
        print(f"Loaded {len(cluster_centers)} cluster centers.")
    except FileNotFoundError:
        print("Error: Clusters not found. You must run train.py (or generate) first!")
        exit()

    # 2. Load Model
    model = GeoguessrModel(num_classes=NUM_CLUSTERS)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
        
    model = model.to(DEVICE)
    model.eval()

    # 3. Create Data Loader
    # mode='eval' ensures we get (img, label, lat, lon) and no shuffling
    val_loader = create_dataloader(
        tar_files=VAL_URLS,
        model_config=model.get_config(),
        cluster_centers=cluster_centers, # Pass numpy array
        batch_size=BATCH_SIZE,
        workers=NUM_WORKERS,
        mode='eval'
    )

    # 4. Inference Loop
    print("Running Inference...")
    correct_clusters = 0
    total_samples = 0
    distances_km = []

    with torch.no_grad():
        for i, (imgs, labels, true_lats, true_lons) in enumerate(val_loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Predict
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            # A. Exact Accuracy
            correct_clusters += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # B. Distance Calculation
            # 1. Get the 3D center of the predicted cluster
            pred_indices = preds.cpu().numpy()
            pred_centers_3d = cluster_centers[pred_indices] 
            
            # 2. Convert predicted 3D -> Lat/Lon
            pred_lats, pred_lons = cartesian_to_latlon(
                pred_centers_3d[:, 0], 
                pred_centers_3d[:, 1], 
                pred_centers_3d[:, 2]
            )
            
            # 3. Calculate Haversine distance vs Truth
            current_lats = true_lats.numpy()
            current_lons = true_lons.numpy()
            
            for j in range(len(pred_lats)):
                dist = haversine_distance(current_lats[j], current_lons[j], pred_lats[j], pred_lons[j])
                distances_km.append(dist)
            
            if (i+1) % 10 == 0:
                print(f"Batch {i+1}: Running Avg Error = {np.mean(distances_km):.1f} km")

    # 5. Final Report
    if total_samples > 0:
        print("\n" + "="*40)
        print(f"   FINAL RESULTS (N={total_samples})   ")
        print("="*40)
        
        accuracy = correct_clusters / total_samples * 100
        mean_dist = np.mean(distances_km)
        median_dist = np.median(distances_km)
        
        print(f"Exact Cluster Acc:     {accuracy:.2f}%")
        print(f"Mean Distance Error:   {mean_dist:.1f} km")
        print(f"Median Distance Error: {median_dist:.1f} km")
        
        # GeoGuessr Scoring (Approximate)
        # 5000pts if < 25m. Drops exponentially.
        score_sum = 0
        for d in distances_km:
            # Simple exponential decay formula similar to GeoGuessr
            score = 5000 * np.exp(-d / 2000) 
            score_sum += score
        avg_score = score_sum / total_samples
        
        print(f"Est. GeoGuessr Score:  {avg_score:.0f} / 5000")
        print("="*40)
    else:
        print("No samples found. Check your URLs or Country List.")