"""
Dataset utilities for OSV5M.

The loader streams tar shards with WebDataset to avoid loading everything
into memory. Each sample is expected to contain:
- "jpg" (or "png") image bytes
- "json" metadata with at least {"lat": float, "lon": float, "country": "US"}
"""
import glob
import json
import os
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import torch
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms

try:
    import webdataset as wds
except ImportError as exc:  # pragma: no cover - hard dependency
    raise ImportError("Install webdataset to use OSV5MDataset: pip install webdataset") from exc

from cluster import Clusterer


def default_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    )


def train_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    )


def _decode_metadata(meta_blob: bytes | str) -> Dict:
    if isinstance(meta_blob, dict):
        return meta_blob
    if isinstance(meta_blob, bytes):
        meta_blob = meta_blob.decode("utf-8")
    return json.loads(meta_blob)


class OSV5MDataset(IterableDataset):
    def __init__(
        self,
        root: str,
        split: str,
        clusterer: Optional[Clusterer],
        countries: Optional[Iterable[str]] = None,
        transform: Optional[transforms.Compose] = None,
        shuffle: int = 2048,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.clusterer = clusterer
        self.countries: Optional[Set[str]] = set(countries) if countries else None
        self.transform = transform or default_image_transform()
        shard_pattern = os.path.join(root, split, "*.tar")
        self.shards: List[str] = sorted(glob.glob(shard_pattern))
        if not self.shards:
            raise FileNotFoundError(f"No shards found at {shard_pattern}")
        self.shuffle = shuffle
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[Dict]:
        dataset = wds.WebDataset(self.shards, resampled=False)
        if self.shuffle:
            dataset = dataset.shuffle(self.shuffle)
        dataset = dataset.decode("pil").to_tuple("jpg;png", "json")
        if self.max_samples:
            dataset = dataset.slice(self.max_samples)

        def filtered() -> Iterator[Dict]:
            for img, meta_blob in dataset:
                meta = _decode_metadata(meta_blob)
                country = meta.get("country")
                if self.countries and country not in self.countries:
                    continue
                lat_val = meta.get("lat") or meta.get("latitude")
                lon_val = meta.get("lon") or meta.get("longitude")
                if lat_val is None or lon_val is None:
                    continue
                lat, lon = float(lat_val), float(lon_val)
                target = meta.get("cluster_id")
                if self.clusterer is not None:
                    target = self.clusterer.assign((lat, lon))
                if target is None:
                    continue
                image: Image.Image = img if isinstance(img, Image.Image) else Image.open(img)
                yield {
                    "image": self.transform(image),
                    "target": int(target),
                    "lat": lat,
                    "lon": lon,
                    "country": country,
                }

        return iter(filtered())


def make_dataloader(
    dataset: IterableDataset,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
