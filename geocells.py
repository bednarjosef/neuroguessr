import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns

def latlon_to_cartesian(df):
    """
    Converts Lat/Lon (degrees) to Cartesian 3D (x, y, z) on a unit sphere.
    This prevents issues with the International Date Line and polar distortion.
    """
    # Convert degrees to radians
    lat_rad = np.deg2rad(df['lat'])
    lon_rad = np.deg2rad(df['lon'])

    # Formula for spherical to cartesian
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.stack([x, y, z], axis=1)

def create_balanced_clusters(df, n_clusters=100):
    """
    Performs K-Means clustering in 3D space.
    K-Means naturally minimizes variance, creating smaller cells in dense areas
    and larger cells in sparse areas, effectively balancing the dataset.
    """
    print(f"Converting {len(df)} points to 3D Cartesian coordinates...")
    coords_3d = latlon_to_cartesian(df)

    # MiniBatchKMeans is faster for large datasets (millions of points)
    print(f"Clustering into {n_clusters} classes...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    labels = kmeans.fit_predict(coords_3d)

    # Assign the cluster label back to the dataframe
    df['class_label'] = labels
    return df, kmeans.cluster_centers_

def visualize_clusters(df):
    """
    Visualizes the Voronoi-like tessellation on a 2D map.
    """
    plt.figure(figsize=(16, 8))
    
    # We use a high-contrast palette to distinguish neighbors
    sns.scatterplot(
        data=df, 
        x='lon', 
        y='lat', 
        hue='class_label', 
        palette='tab20', 
        s=10, 
        legend=False,
        linewidth=0
    )

    plt.title("Adaptive Geocells (Voronoi Tessellation)", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    
    print("Displaying map... (Close window to finish)")
    plt.show()

def generate_dummy_data(n_samples=20000):
    """
    Generates clumped data to simulate real Street View coverage.
    (Dense in Europe/US, sparse elsewhere).
    """
    print("Generating dummy 'clumped' data...")
    
    # Centers of "dense" areas (Europe, US East, Japan)
    centers = [
        (48.85, 2.35),   # Paris
        (40.71, -74.00), # NYC
        (35.68, 139.69), # Tokyo
        (51.50, -0.12),  # London
        (-33.86, 151.20) # Sydney (Sparse surrounding)
    ]
    
    lats, lons = [], []
    
    # Generate clusters
    for lat, lon in centers:
        # Create Gaussian blobs around cities
        lats.append(np.random.normal(lat, 5, n_samples // len(centers)))
        lons.append(np.random.normal(lon, 5, n_samples // len(centers)))
        
    # Add some random world noise (rural areas)
    lats.append(np.random.uniform(-60, 80, n_samples // 10))
    lons.append(np.random.uniform(-180, 180, n_samples // 10))
    
    lats = np.concatenate(lats)
    lons = np.concatenate(lons)
    
    return pd.DataFrame({'lat': lats, 'lon': lons})

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data (Replace this with your pd.read_csv('data.csv'))
    df = generate_dummy_data(n_samples=50000)

    # 2. Create the Clusters
    # n_clusters is the number of output neurons your AI will have
    df_clustered, centers = create_balanced_clusters(df, n_clusters=50)

    # 3. Check Balance
    counts = df_clustered['class_label'].value_counts()
    print("\n--- Balance Report ---")
    print(f"Standard Deviation of Class Sizes: {counts.std():.2f}")
    print(f"Max Class Size: {counts.max()}")
    print(f"Min Class Size: {counts.min()}")
    
    # 4. Visualize
    visualize_clusters(df_clustered)
    