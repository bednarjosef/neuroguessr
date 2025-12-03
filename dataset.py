import os, glob
import torch
from torch.utils.data import IterableDataset, DataLoader
import webdataset as wds
from datasets import load_dataset

from clusters import get_closest_cluster


class OSVDataset(IterableDataset):
    def __init__(self, countries, tar_directory, cluster_centers, transform):
        self.countries = countries
        self.tar_directory = tar_directory
        self.tar_files = self._find_files()
        self.cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32)
        self.transform = transform

    def _find_files(self):
        search_path = os.path.join(self.tar_directory, "*.tar")
        tar_files = glob.glob(search_path)
        return tar_files

    def __iter__(self):
        dataset = wds.WebDataset(self.tar_files, resampled=True, shardshuffle=100, handler=wds.warn_and_continue).shuffle(10000).decode('pil').to_tuple('jpg', 'json')
        for img, meta in dataset:
            try:
                country = meta.get('country')
                if not country or country not in self.countries:
                    continue

                lat, lon = meta.get('latitude'), meta.get('longitude')
                if lat is None or lon is None:
                    continue

                cluster_label = get_closest_cluster(lat, lon, self.cluster_centers)
                
                img_tensor = self.transform(img.convert("RGB"))
                yield img_tensor, cluster_label, lat, lon
            
            # simply skip if any error
            except Exception:
                continue


class StreetViewDataset(IterableDataset):
    def __init__(self, repo_id, countries, cluster_centers, transform, split="train"):
        self.repo_id = repo_id
        self.countries = set(countries)
        self.cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32)
        self.transform = transform
        self.split = split
    
        self.dataset = None

    def _init_dataset(self):
        ds = load_dataset(self.repo_id, split=self.split)
        ds = ds.to_iterable_dataset(num_shards=393)
        ds = ds.shuffle(seed=42, buffer_size=10_000)
        return ds

    def __iter__(self):
        if self.dataset is None:
            self.dataset = self._init_dataset()

        # worker_info = torch.utils.data.get_worker_info()
        if self.dataset is None:
            self.dataset = self._init_dataset()

        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is not None:
        #     iterator = iter(self.dataset)
        # else:
        #     iterator = iter(self.dataset)

        for sample in self.dataset:
            try:
                country = sample.get('country_code')
                if not country or country not in self.countries:
                    continue

                lat, lon = sample.get('latitude'), sample.get('longitude')
                if lat is None or lon is None:
                    continue

                cluster_label = get_closest_cluster(lat, lon, self.cluster_centers)                
                img_tensor = self.transform(sample['image'].convert("RGB"))
                
                yield img_tensor, cluster_label, torch.tensor(lat), torch.tensor(lon)

            except Exception as e:
                print(f"Skipping bad {self.split} sample: {e}") 
                continue


def create_osv_dataloader(CONFIG, tar_directory, cluster_centers, transform, workers):
    dataset = OSVDataset(CONFIG['countries'], tar_directory, cluster_centers, transform)
    loader = DataLoader(dataset, CONFIG['batch_size'], num_workers=workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    return loader


def create_streetview_dataloader(CONFIG, ds_dir, split, cluster_centers, transform, workers):
    dataset = StreetViewDataset(ds_dir, CONFIG['countries'], cluster_centers, transform, split)
    loader = DataLoader(dataset, CONFIG['batch_size'], num_workers=workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    return loader
