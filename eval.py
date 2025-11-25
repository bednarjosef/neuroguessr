import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds
from model import GeoguessrModel
import numpy as np
import math

# --- CONFIGURATION ---
MODEL_PATH = "geoguessr_geocells_200.pth"
CLUSTER_PATH = "cluster_centers_200.npy" # You must have this file!
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLUSTERS = 200

# Target Countries (Used for filtering input data only)
TARGET_COUNTRIES = [
    'AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG', 'KH', 'CA', 'CL', 'CO', 'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI', 'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS', 'IN', 'ID', 'IE', 'IL', 'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY', 'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA', 'PE', 'PH', 'PL', 'PT', 'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS', 'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE', 'CH', 'TW', 'TH', 'TR', 'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN',
]
country_set = set(TARGET_COUNTRIES)

# --- MATH HELPER FUNCTIONS ---
def cartesian_to_latlon(x, y, z):
    """Converts 3D cartesian coordinates back to Lat/Lon (degrees)."""
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

# --- DATASET CLASS ---
class WebDatasetGeocellEval(IterableDataset):
    def __init__(self, urls, transform, target_countries, cluster_centers):
        self.urls = urls
        self.transform = transform
        self.target_countries = target_countries
        self.cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32)

    def get_closest_cluster(self, lat, lon):
        # Convert single point to 3D
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        point = torch.tensor([x, y, z], dtype=torch.float32)
        
        # Find nearest center
        dists = torch.sum((self.cluster_centers - point)**2, dim=1)
        return torch.argmin(dists).item()

    def __iter__(self):
        dataset = (
            wds.WebDataset(self.urls, resampled=False, handler=wds.warn_and_continue)
            .decode("pil")
            .to_tuple("jpg", "json")
        )

        for img, meta in dataset:
            try:
                # Filter by country
                country = meta.get('country')
                if not country or country not in self.target_countries:
                    continue
                
                lat = meta.get('latitude')
                lon = meta.get('longitude')
                if lat is None or lon is None:
                    continue

                # Generate Ground Truth
                label = self.get_closest_cluster(lat, lon)
                img_tensor = self.transform(img.convert("RGB"))
                
                # Yield: Image, Class Label, Actual Lat, Actual Lon
                yield img_tensor, label, lat, lon
            except Exception:
                continue

# --- MAIN EVALUATION ---
if __name__ == "__main__":
    print(f"Evaluating on device: {DEVICE}")

    # 1. Load Cluster Centers (Required for Distance Calculation)
    try:
        # Load the 3D coordinates (200, 3)
        cluster_centers_3d = np.load(CLUSTER_PATH)
        print(f"Loaded {len(cluster_centers_3d)} cluster centers.")
    except Exception as e:
        print(f"Error loading {CLUSTER_PATH}: {e}")
        print("Did you run train.py? It saves this file at the end.")
        exit()

    # 2. Load Model
    # Note: num_classes is now NUM_CLUSTERS (200), not len(countries)
    model = GeoguessrModel(num_classes=NUM_CLUSTERS)
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit()
    
    model = model.to(DEVICE)
    model.eval()

    # 3. Setup Data
    data_config = model.get_config()
    transforms = timm.data.create_transform(**data_config, is_training=False)

    print("Generating Validation URLs...")
    # Using the test split from OSV5M
    base_url = "https://huggingface.co/datasets/osv5m/osv5m-wds/resolve/main/test/{:04d}.tar"
    urls = [base_url.format(i) for i in range(10)] # Limit to 10 shards for quick test

    val_dataset = WebDatasetGeocellEval(urls, transforms, country_set, cluster_centers_3d)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # 4. Run Inference
    print("Starting Inference...")
    
    correct_clusters = 0
    total_samples = 0
    distances_km = []
    
    with torch.no_grad():
        for i, (imgs, labels, true_lats, true_lons) in enumerate(val_loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward Pass
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            # --- METRICS ---
            # 1. Classification Accuracy (Did we pick the exact right cell?)
            correct_clusters += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # 2. Distance Error (The GeoGuessr Metric)
            # Convert predictions (cluster IDs) back to Lat/Lon
            pred_indices = preds.cpu().numpy()
            
            # Get the 3D center of the predicted cluster
            pred_centers_3d = cluster_centers_3d[pred_indices] 
            
            # Convert 3D -> Lat/Lon
            pred_lats, pred_lons = cartesian_to_latlon(
                pred_centers_3d[:, 0], 
                pred_centers_3d[:, 1], 
                pred_centers_3d[:, 2]
            )
            
            # Calculate distance for batch
            current_lats = true_lats.numpy()
            current_lons = true_lons.numpy()
            
            for j in range(len(pred_lats)):
                dist = haversine_distance(current_lats[j], current_lons[j], pred_lats[j], pred_lons[j])
                distances_km.append(dist)
            
            if (i+1) % 10 == 0:
                avg_dist_so_far = sum(distances_km) / len(distances_km)
                print(f"Batch {i+1}: Avg Error = {avg_dist_so_far:.1f} km")

    # 5. Final Report
    print("\n" + "="*40)
    print("   EVALUATION RESULTS (GEOCELLS)   ")
    print("="*40)
    
    if total_samples > 0:
        accuracy = correct_clusters / total_samples * 100
        median_dist = np.median(distances_km)
        mean_dist = np.mean(distances_km)
        
        # Calculate "2500km Score" (GeoGuessr standard: 5000pts if <25m, 0pts if >2000km approx)
        # This is a loose approximation
        guesses_under_100km = sum(1 for d in distances_km if d < 100)
        percentage_under_100km = (guesses_under_100km / total_samples) * 100
        
        print(f"Total Images:          {total_samples}")
        print(f"Exact Cluster Acc:     {accuracy:.2f}%")
        print(f"Mean Distance Error:   {mean_dist:.1f} km")
        print(f"Median Distance Error: {median_dist:.1f} km")
        print(f"Guesses < 100km:       {percentage_under_100km:.1f}%")
        
        print("-" * 40)
        if mean_dist < 1500:
            print("Verdict: The model is learning geography!")
        elif mean_dist < 3000:
            print("Verdict: Model understands continents, but not regions.")
        else:
            print("Verdict: Model is guessing randomly.")
    else:
        print("No valid images found.")