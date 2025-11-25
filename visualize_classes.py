import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
TARGET_COUNTRIES = [
    'AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG', 'KH', 'CA', 'CL', 'CO', 'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI', 'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS', 'IN', 'ID', 'IE', 'IL', 'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY', 'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA', 'PE', 'PH', 'PL', 'PT', 'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS', 'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE', 'CH', 'TW', 'TH', 'TR', 'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN',
]
NUM_CLUSTERS = 200  # How many total geocells you want across these countries
REPO_ID = "osv5m/osv5m" 
FILENAME = "train.csv"

# --- 1. DATA LOADING ---
print(f"Downloading/Loading {FILENAME} from {REPO_ID}...")
try:
    csv_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
except Exception as e:
    print(f"Error downloading: {e}")
    exit()

print("Reading CSV (loading coords and country)...")
# Note: OSV5M uses 'latitude' and 'longitude'
df = pd.read_csv(csv_path, usecols=['country', 'latitude', 'longitude'])

# Filter for Target Countries
print(f"Filtering for {len(TARGET_COUNTRIES)} countries: {TARGET_COUNTRIES}")
df_subset = df[df['country'].isin(TARGET_COUNTRIES)].copy()
df_subset = df_subset.dropna(subset=['latitude', 'longitude'])

print(f"Total training points found: {len(df_subset):,}")

# --- 2. CLUSTERING LOGIC (3D) ---
def latlon_to_cartesian(lat, lon):
    """Converts lat/lon to 3D cartesian coordinates on a unit sphere."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=1)

print("Converting coordinates to 3D space...")
coords_3d = latlon_to_cartesian(df_subset['latitude'].values, df_subset['longitude'].values)

print(f"Running K-Means to generate {NUM_CLUSTERS} Geocells...")
# MiniBatchKMeans is much faster for large datasets like OSV5M
kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, random_state=42, batch_size=4096)
df_subset['cluster_label'] = kmeans.fit_predict(coords_3d)

# --- 3. VISUALIZATION ---
print("Generating Map Visualization...")

plt.figure(figsize=(16, 8), facecolor='#111111')
ax = plt.gca()
ax.set_facecolor('#111111')

# We sample the data for the plot to keep it fast (plotting 1M+ points is slow)
plot_sample = df_subset.sample(min(100000, len(df_subset)))

# Plot the points, colored by their cluster
# We use a qualitative palette to make distinct clusters visible
sns.scatterplot(
    data=plot_sample,
    x='longitude',
    y='latitude',
    hue='cluster_label',
    palette='turbo', # 'turbo' or 'tab20' provides good high-contrast separation
    s=5,
    linewidth=0,
    legend=False,
    alpha=0.7
)

# Plot the cluster centers (Centroids)
# We need to convert 3D centers back to Lat/Lon to plot them
centers_3d = kmeans.cluster_centers_
# Inverse formula:
center_lats = np.rad2deg(np.arcsin(centers_3d[:, 2]))
center_lons = np.rad2deg(np.arctan2(centers_3d[:, 1], centers_3d[:, 0]))

plt.scatter(
    center_lons, 
    center_lats, 
    c='white', 
    s=30, 
    marker='x', 
    alpha=0.8, 
    label='Cluster Centers'
)

plt.title(f"Voronoi Geocells for Selected Countries (K={NUM_CLUSTERS})", color='white', fontsize=16)
plt.xlabel("Longitude", color='white')
plt.ylabel("Latitude", color='white')
plt.tick_params(colors='white')
plt.grid(True, color='#333333', alpha=0.5)
plt.xlim(-180, 180)
plt.ylim(-90, 90)

# Text info on the plot
info_text = f"Dataset: {len(df_subset):,}" # images\nCountries: {', '.join(TARGET_COUNTRIES)}
plt.text(-175, -85, info_text, color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.7))

print("Displaying plot...")
plt.tight_layout()
plt.show()

# --- 4. STATISTICS ---
print("\n--- Cluster Distribution ---")
# How many clusters end up in each country? (Approximate based on majority vote per cluster)
# This helps you see if 'France' dominates 'Japan' in terms of cluster count.
cluster_counts = df_subset['cluster_label'].value_counts()
print(f"Largest Cluster Size: {cluster_counts.max()} images")
print(f"Smallest Cluster Size: {cluster_counts.min()} images")
print(f"Average images per cluster: {cluster_counts.mean():.0f}")
