import torch.nn.functional as F
import torch

from clusters import xyz_to_latlon

EARTH_RADIUS = 6371.0


def haversine_batch(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return EARTH_RADIUS * c


class PIGEONLoss():
    def __init__(self, CONFIG, classifier):

        x = classifier.CLASS_CENTERS_XYZ[:, 0]
        y = classifier.CLASS_CENTERS_XYZ[:, 1]
        z = classifier.CLASS_CENTERS_XYZ[:, 2]

        lats, lons = xyz_to_latlon(x, y, z)  # numpy
        self.cluster_lats = torch.tensor(lats, dtype=torch.float32, device=CONFIG['device'])
        self.cluster_lons = torch.tensor(lons, dtype=torch.float32, device=CONFIG['device'])

    def loss(self, CONFIG, logits, true_clusters, true_lats_deg, true_lons_deg):
        """
        logits: (B, K) model outputs
        true_clusters: (B,) long, index of correct geocell/cluster
        true_lats_deg, true_lons_deg: (B,) in degrees
        cluster_lats, cluster_lons: (K,) in degrees
        """

        # convert to radians & shape for broadcasting
        lat1 = torch.deg2rad(true_lats_deg).unsqueeze(1)      # (B, 1)
        lon1 = torch.deg2rad(true_lons_deg).unsqueeze(1)      # (B, 1)
        lat2 = torch.deg2rad(self.cluster_lats).unsqueeze(0)       # (1, K)
        lon2 = torch.deg2rad(self.cluster_lons).unsqueeze(0)       # (1, K)

        # distances from each sample to each cluster center
        dists = haversine_batch(lat1, lon1, lat2, lon2)       # (B, K)

        # distance to true cell (per sample)
        d_true = dists.gather(1, true_clusters.unsqueeze(1))  # (B, 1)

        # weights y_{n,i} = exp(-(d_i - d_true)/tau)
        weights = torch.exp(-(dists - d_true) / CONFIG['tau_km'])          # (B, K)

        # OPTIONAL: normalize per row (not required mathematically, but common)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        log_probs = F.log_softmax(logits, dim=1)
        loss_per_sample = -(weights * log_probs).sum(dim=1)
        return loss_per_sample.mean()
