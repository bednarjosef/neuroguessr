import os, glob
import torch
from torch.utils.data import IterableDataset, DataLoader
import webdataset as wds

from clusters import get_closest_cluster


class ClusterDataset(IterableDataset):
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
        dataset = wds.WebDataset(self.tar_files, shardshuffle=100, handler=wds.warn_and_continue).shuffle(10000).decode('pil').to_tuple('jpg', 'json')
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
                yield img_tensor, cluster_label
            
            # simply skip if any error
            except Exception:
                continue
        

def create_dataloader(CONFIG, tar_directory, cluster_centers, transform, workers):
    dataset = ClusterDataset(CONFIG['countries'], tar_directory, cluster_centers, transform)
    loader = DataLoader(dataset, CONFIG['batch_size'], num_workers=workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    return loader
