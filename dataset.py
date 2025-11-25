import datetime
import torch
from torch.utils.data import IterableDataset, DataLoader
import webdataset as wds
import numpy as np
import timm
from clusters import TARGET_COUNTRIES
from torchvision import transforms

from torchvision import transforms

def get_geo_transforms(model_config, is_training=True):
    mean = model_config['mean']
    std = model_config['std']
    
    if is_training:
        return transforms.Compose([
            # 1. Resize small edge to 256 (keep aspect ratio)
            transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
            # 2. Random Crop (But conservative! 80% to 100% of image)
            # This prevents the "Zoomed in Grass" problem
            transforms.RandomResizedCrop(
                size=(384, 384), 
                scale=(0.8, 1.0), # <--- CRITICAL FIX: Only crop large chunks
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            # 3. Color Jitter (Helpful for weather generalization)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Eval: Standard Center Crop
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

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
                cluster_label = self.get_closest_cluster(lat, lon)
                climate_label = int(meta.get('climate', -1))
                land_label = int(meta.get('land_cover', -1))
                soil_label = int(meta.get('soil', -1))

                if climate_label < 0 or climate_label >= 31: 
                    climate_label = -1

                if land_label < 0 or land_label >= 11:
                    land_label = -1

                if soil_label < 0 or soil_label >= 15:
                    soil_label = -1
                
                ts = meta.get('captured_at')
                if ts:
                    month_label = datetime.datetime.fromtimestamp(ts / 1000.0).month - 1
                else:
                    month_label = -1

                img_tensor = self.transform(img.convert("RGB"))
                
                if self.mode == 'train':
                    yield img_tensor, cluster_label, climate_label, land_label, soil_label, month_label
                else:
                    yield img_tensor, cluster_label, climate_label, land_label, soil_label, month_label
                    
            except Exception:
                continue

def create_dataloader(tar_files, model_config, cluster_centers, batch_size, workers, mode='train', custom_transform=None):
    if custom_transform:
        transforms = custom_transform
    else:
        # Fallback to the default broken one (don't use this path)
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