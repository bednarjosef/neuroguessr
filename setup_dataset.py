import pandas as pd
import numpy as np
import os
from huggingface_hub import hf_hub_download
from sklearn.cluster import MiniBatchKMeans
from datasets import load_dataset


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
    return centers, labels, df


if __name__ == '__main__':
    countries = ['AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG', 'KH', 'CA', 'CL', 'CO', 'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI', 'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS', 'IN', 'ID', 'IE', 'IL', 'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY', 'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA', 'PE', 'PH', 'PL', 'PT', 'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS', 'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE', 'CH', 'TW', 'TH', 'TR', 'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN']
    centers, labels, df = generate_clusters(countries, 256, 'cache')


    num_clusters = 256
    df["cluster"] = labels
    max_per_cluster = 1000

    sampled_frames = []
    cluster_counts = {}

    rng = np.random.default_rng(42)

    for c in range(num_clusters):
        print(f'gathering for cluster {c}')
        g = df[df["cluster"] == c]
        n = len(g)
        cluster_counts[c] = n

        if n == 0:
            print(f"cluster {c}: 0 samples in CSV")
            continue

        if n <= max_per_cluster:
            sampled = g
        else:
            # random choice of indices inside this cluster
            idx = rng.choice(g.index.values, size=max_per_cluster, replace=False)
            sampled = g.loc[idx]

        sampled_frames.append(sampled)

    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    print("Total sampled rows:", len(sampled_df))        
