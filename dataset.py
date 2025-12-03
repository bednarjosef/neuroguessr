import os, glob
import torch
from torch.utils.data import IterableDataset, DataLoader
import webdataset as wds
from datasets import load_dataset
from itertools import islice  # <--- NEW IMPORT

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
        # OSV uses WebDataset, which handles sharding automatically if tar_files list is passed correctly
        # Usually checking worker_info here implies splitting tar_files list manually, 
        # but wds.WebDataset often handles this if nodesplitter is default. 
        # For safety/simplicity given your previous code, we keep this as is:
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

        self.kept_count = 0
        self.dropped_count = 0

    def _init_dataset(self):
        return load_dataset(self.repo_id, split=self.split, streaming=True)

    def __iter__(self):
        if self.dataset is None:
            self.dataset = self._init_dataset()

        worker_info = torch.utils.data.get_worker_info()
    
        shuffled_ds = self.dataset.shuffle(seed=42, buffer_size=10_000)
        
        iterator = iter(shuffled_ds)

        if worker_info is not None:
            iterator = islice(iterator, worker_info.id, None, worker_info.num_workers)

        for sample in iterator:
            try:
                total = self.kept_count + self.dropped_count
                if total > 0 and total % 1000 == 0:
                    print(f"Worker stats: Kept {self.kept_count} | Dropped {self.dropped_count} | Pass Rate: {self.kept_count/total:.1%}")

                country = sample.get('country_code')
                if not country or country not in self.countries:
                    self.dropped_count += 1
                    continue

                lat, lon = sample.get('latitude'), sample.get('longitude')
                if lat is None or lon is None:
                    continue
                
                self.kept_count += 1

                cluster_label = get_closest_cluster(lat, lon, self.cluster_centers)                
                img_tensor = self.transform(sample['image'].convert("RGB"))
                
                yield img_tensor, cluster_label, torch.tensor(lat), torch.tensor(lon)

            except Exception as e:
                # Optional: reduce verbosity if too many errors
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
