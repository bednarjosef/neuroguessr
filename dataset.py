import torch
from torch.utils.data import IterableDataset, DataLoader
import webdataset as wds
import numpy as np
import timm
from clusters import TARGET_COUNTRIES

class ClusterDataset(IterableDataset):
    def __init__(self, tar_paths, transform, cluster_centers, mode='train'):
        self.tar_paths = tar_paths
        self.transform = transform
        self.mode = mode
        self.target_countries = set(TARGET_COUNTRIES)
        
        self.cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32)

    def get_closest_cluster(self, lat, lon):

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        
        point = torch.tensor([x, y, z], dtype=torch.float32)
        
        dists = torch.sum((self.cluster_centers - point)**2, dim=1)
        return torch.argmin(dists).item()

    def __iter__(self):
        if self.mode == 'train':
            dataset = (
                wds.WebDataset(self.tar_paths, resampled=True, shardshuffle=True, handler=wds.warn_and_continue)
                .shuffle(10000)
                .decode("pil")
                .to_tuple("jpg", "json")
            )
        else:
            # Eval: Single pass (resampled=False), No shuffling
            dataset = (
                wds.WebDataset(self.tar_paths, resampled=False, handler=wds.warn_and_continue)
                .decode("pil")
                .to_tuple("jpg", "json")
            )

        for img, meta in dataset:
            try:
                # 1. Filter by Country
                country = meta.get('country')
                if not country or country not in self.target_countries:
                    continue
                
                # 2. Get Coordinates
                lat = meta.get('latitude')
                lon = meta.get('longitude')
                if lat is None or lon is None:
                    continue

                # 3. Calculate Label (The heavy lifting)
                label = self.get_closest_cluster(lat, lon)

                # 4. Transform Image
                img_tensor = self.transform(img.convert("RGB"))
                
                # 5. Yield based on mode
                if self.mode == 'train':
                    yield img_tensor, label
                else:
                    # For Eval, we need the raw lat/lon to calculate km error later
                    yield img_tensor, label, lat, lon
                    
            except Exception:
                continue

def create_dataloader(tar_files, model_config, cluster_centers, batch_size, workers, mode='train'):
    is_training = (mode == 'train')
    transforms = timm.data.create_transform(**model_config, is_training=is_training)
    
    dataset = ClusterDataset(
        tar_paths=tar_files,
        transform=transforms,
        cluster_centers=cluster_centers,
        mode=mode
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=4 if workers > 0 else None,
        persistent_workers=(workers > 0)
    )
    
    return loader