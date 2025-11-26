import os
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
