"""
Geospatial clustering utilities.

Uses MiniBatchKMeans over 3D Cartesian coordinates on the unit sphere to
obtain roughly uniform geographic clusters. Artifacts are saved to NPZ so
training/eval can reuse cluster centroids without recomputing.
"""
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from huggingface_hub import hf_hub_download

EARTH_RADIUS_KM = 6371.0088


def _latlon_to_xyz(lat: float, lon: float) -> np.ndarray:
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    return np.array(
        [math.cos(lat_r) * math.cos(lon_r), math.cos(lat_r) * math.sin(lon_r), math.sin(lat_r)],
        dtype=np.float32,
    )


def _xyz_to_latlon(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz
    lon = math.atan2(y, x)
    hyp = math.hypot(x, y)
    lat = math.atan2(z, hyp)
    return np.array([math.degrees(lat), math.degrees(lon)], dtype=np.float32)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


class Clusterer:
    def __init__(self, centroids_latlon: np.ndarray):
        self.centroids_latlon = np.asarray(centroids_latlon, dtype=np.float32)
        self.centroids_xyz = np.stack([_latlon_to_xyz(lat, lon) for lat, lon in self.centroids_latlon])

    def assign(self, coord: Tuple[float, float]) -> int:
        xyz = _latlon_to_xyz(coord[0], coord[1])
        dots = self.centroids_xyz @ xyz
        return int(np.argmax(dots))

    def centroid(self, idx: int) -> Tuple[float, float]:
        lat, lon = self.centroids_latlon[idx]
        return float(lat), float(lon)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, centroids_latlon=self.centroids_latlon)

    @classmethod
    def load(cls, path: str | Path) -> "Clusterer":
        data = np.load(path)
        return cls(data["centroids_latlon"])

    def distance_to_center(self, coord: Tuple[float, float], idx: int) -> float:
        lat, lon = coord
        center_lat, center_lon = self.centroid(idx)
        return haversine_km(lat, lon, center_lat, center_lon)


def fit_clusters(
    coords: Iterable[Tuple[float, float]],
    n_clusters: int,
    batch_size: int = 4096,
    max_iter: int = 200,
    seed: int = 17,
    out_path: str | Path = "artifacts/clusters.npz",
) -> Clusterer:
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        verbose=1,
        n_init="auto",
        random_state=seed,
    )
    buffer: List[np.ndarray] = []
    for lat, lon in coords:
        buffer.append(_latlon_to_xyz(lat, lon))
        if len(buffer) >= batch_size:
            km.partial_fit(np.stack(buffer))
            buffer.clear()
    if buffer:
        km.partial_fit(np.stack(buffer))

    centroids_latlon = np.stack([_xyz_to_latlon(c) for c in km.cluster_centers_])
    clusterer = Clusterer(centroids_latlon)
    clusterer.save(out_path)
    return clusterer


def iter_coords_from_jsonl(path: str | Path) -> Iterator[Tuple[float, float]]:
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            yield float(row["lat"]), float(row["lon"])


def iter_coords_from_csv(path: str | Path) -> Iterator[Tuple[float, float]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = row.get("lat") or row.get("latitude")
            lon = row.get("lon") or row.get("longitude")
            if lat is None or lon is None:
                continue
            yield float(lat), float(lon)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Fit geographic clusters from metadata.")
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Local path to JSONL/CSV containing lat/lon fields. If omitted, train.csv is fetched from osv5m/osv5m.",
    )
    parser.add_argument("--n_clusters", type=int, default=1024, help="Number of geographic clusters.")
    parser.add_argument("--batch_size", type=int, default=4096, help="MiniBatchKMeans batch size.")
    parser.add_argument("--max_iter", type=int, default=200, help="KMeans iterations.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument("--out", type=str, default="artifacts/clusters.npz", help="Destination artifact path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.metadata is None:
        csv_path = hf_hub_download(repo_id="osv5m/osv5m", filename="train.csv", repo_type="dataset")
        coords = iter_coords_from_csv(csv_path)
    elif args.metadata.endswith(".csv"):
        coords = iter_coords_from_csv(args.metadata)
    else:
        coords = iter_coords_from_jsonl(args.metadata)
    fit_clusters(
        coords=coords,
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        seed=args.seed,
        out_path=args.out,
    )
    print(f"Cluster centroids saved to {args.out}")
