"""
Evaluation utilities for GeoGuessr model.
"""
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import nn

from cluster import Clusterer


def geo_score(distance_km: float, scale_km: float = 2000.0) -> float:
    return float(np.exp(-distance_km / scale_km) * 5000.0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: Iterable,
    clusterer: Clusterer,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total, correct = 0, 0
    distances: List[float] = []
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        lat = batch["lat"].float()
        lon = batch["lon"].float()

        logits, _ = model(images)
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()

        preds_np = preds.cpu().numpy()
        lat_np = lat.cpu().numpy()
        lon_np = lon.cpu().numpy()
        for idx, p in enumerate(preds_np):
            dist = clusterer.distance_to_center((float(lat_np[idx]), float(lon_np[idx])), int(p))
            distances.append(dist)

    acc = correct / max(total, 1)
    distances_np = np.array(distances, dtype=np.float32) if distances else np.zeros(1, dtype=np.float32)
    metrics = {
        "top1": acc,
        "median_km": float(np.median(distances_np)),
        "mean_km": float(np.mean(distances_np)),
        "geo_score": float(np.mean([geo_score(d) for d in distances_np])) if len(distances_np) else 0.0,
    }
    return metrics
