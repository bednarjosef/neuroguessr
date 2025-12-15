import os, torch
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from huggingface_hub import hf_hub_download


def latlon_to_xyz(lat, lon):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    if isinstance(lat, np.ndarray):
        return np.stack([x, y, z], axis=1)
    return x, y, z


def xyz_to_latlon(x, y, z):
    lats = np.degrees(np.arcsin(z))
    lons = np.degrees(np.arctan2(y, x))
    return lats, lons


def generate_clusters(countries, num_clusters, cache_dir):
    # Download/load dataset
    csv_path = hf_hub_download(repo_id="osv5m/osv5m", filename="train.csv", repo_type="dataset")
    df = pd.read_csv(csv_path, usecols=['country', 'latitude', 'longitude'])

    # Filter for our countries
    df = df[df['country'].isin(countries)].dropna()

    # Convert to XYZ
    coords_xyz = latlon_to_xyz(df['latitude'].values, df['longitude'].values)

    # Generate clusters
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=4096)
    labels = kmeans.fit_predict(coords_xyz)
    centers = kmeans.cluster_centers_

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(f'{cache_dir}/centers_{num_clusters}.npy', centers)
    return centers


def load_clusters(num_clusters, cache_dir):
    centers_path = f'{cache_dir}/centers_{num_clusters}.npy'
    if not os.path.exists(centers_path):
        raise FileNotFoundError(f'centers_{num_clusters}.npy not found in cache ({cache_dir}).')

    centers = np.load(centers_path)
    return centers


def get_clusters(CONFIG):
    try:
        cluster_centers = load_clusters(CONFIG['clusters'], CONFIG['cache_dir'])
    except FileNotFoundError:
        cluster_centers = generate_clusters(CONFIG['countries'], CONFIG['clusters'], CONFIG['cache_dir'])
    return cluster_centers


def get_cluster_labels(CONFIG, centers):    
    z = centers[:, 2]
    y = centers[:, 1]
    x = centers[:, 0]

    lats, lons = xyz_to_latlon(x, y, z)
    
    # pairwise haversine distances (N x N)
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    
    lat1 = lat_rad[:, None]
    lat2 = lat_rad[None, :]
    lon1 = lon_rad[:, None]
    lon2 = lon_rad[None, :]
    
    dphi = lat2 - lat1
    dlambda = lon2 - lon1
    
    a = np.sin(dphi/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dists_km = 6371 * c
    
    # apply gaussian kernel
    # P(x) = exp(-dist^2 / (2 * sigma^2))
    targets = np.exp(-(dists_km**2) / (2 * CONFIG['sigma_km']**2))
    
    # normalize so rows sum to 1.0 (Probability Distribution)
    sums = targets.sum(axis=1, keepdims=True)
    soft_targets = targets / sums
    
    # convert to Tensor (N, N)
    return torch.tensor(soft_targets, dtype=torch.float32)


def get_closest_cluster(lat, lon, cluster_centers):
    x, y, z = latlon_to_xyz(lat, lon)
    point = torch.tensor([x, y, z], dtype=torch.float32)
    
    dists = torch.sum((cluster_centers - point)**2, dim=1)
    return torch.argmin(dists).item()


## H3 UTILS
# with open(MAPPING_PATH) as f:
#     _raw = json.load(f)


# H3_TO_CLASS = {cell: int(cls) for cell, cls in _raw.items()}

# NUM_CLASSES = max(H3_TO_CLASS.values()) + 1

# with open(COUNTS_PATH) as f:
#     _counts = json.load(f)

# COUNTS = {cell: int(c) for cell, c in _counts.items() if cell in H3_TO_CLASS}

# import math
# import numpy as np

# # number of classes from the mapping
# NUM_CLASSES = max(H3_TO_CLASS.values()) + 1

# # we do a *count-weighted* average of cell centers, then re-normalize to unit length
# sum_vec = np.zeros((NUM_CLASSES, 3), dtype=np.float64)
# sum_w = np.zeros(NUM_CLASSES, dtype=np.float64)

# for cell, cls in H3_TO_CLASS.items():
#     # weight = how many images in this cell
#     w = COUNTS.get(cell, 1)

#     # center lat/lon of this H3 cell
#     lat, lon = h3.cell_to_latlng(cell)
#     lat_r = math.radians(float(lat))
#     lon_r = math.radians(float(lon))

#     # convert to unit sphere xyz
#     x = math.cos(lat_r) * math.cos(lon_r)
#     y = math.cos(lat_r) * math.sin(lon_r)
#     z = math.sin(lat_r)

#     # accumulate weighted vector
#     sum_vec[cls, 0] += x * w
#     sum_vec[cls, 1] += y * w
#     sum_vec[cls, 2] += z * w
#     sum_w[cls] += w

# centers = np.zeros_like(sum_vec)
# for i in range(NUM_CLASSES):
#     if sum_w[i] > 0:
#         v = sum_vec[i] / sum_w[i]
#         norm = np.linalg.norm(v)
#         if norm > 0:
#             centers[i] = v / norm
#         else:
#             centers[i] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
#     else:
#         centers[i] = np.array([1.0, 0.0, 0.0], dtype=np.float64)

# CLASS_CENTERS_XYZ = centers.astype(np.float32)  # shape: [num_classes, 3]


# def class_id_to_latlon(class_id: int):
#     """Representative center (lat, lon) for a class."""
#     x, y, z = CLASS_CENTERS_XYZ[class_id]
#     lat = math.degrees(math.asin(z))
#     lon = math.degrees(math.atan2(y, x))
#     return lat, lon


# def latlon_to_class(lat, lon, default=-1):
#     cell = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
#     return H3_TO_CLASS.get(cell, default)


# if __name__ == '__main__':
#     print(len(H3_TO_CLASS))   # should be 1999060 cells mapped or close
#     print(max(H3_TO_CLASS.values()))  # should be 2121 (0..2121)

#     lat, lon = 50.087, 14.421  # Prague test
#     print(latlon_to_class(lat, lon))


#     print("num classes:", NUM_CLASSES)            # should be 2122
#     print(CLASS_CENTERS_XYZ.shape)                # (2122, 3)

#     lat, lon = 15.327128, -14.897281   # Prague
#     cid = latlon_to_class(lat, lon)
#     print("Prague class:", cid)
#     print("center of that class:", class_id_to_latlon(cid))
    